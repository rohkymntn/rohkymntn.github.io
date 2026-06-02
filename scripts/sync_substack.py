#!/usr/bin/env python3
"""Sync a Substack publication into posts.json (the site's structure).

Why this is a two-step fetch:
  * The RSS feed is used only to ENUMERATE posts (title, link, date).
  * Each post's BODY is pulled from its LIVE WEB PAGE, because Substack's RSS
    `content:encoded` is frozen near publish time and does NOT reflect later
    edits to an already-published post. The web page always has current text.

Stdlib only (no pip installs). Usage:
    python3 scripts/sync_substack.py [feed_url] [out_path]
"""
import sys, re, json, urllib.request
from html import escape, unescape
from html.parser import HTMLParser
from email.utils import parsedate_to_datetime

FEED_URL = sys.argv[1] if len(sys.argv) > 1 else "https://kyleroh.substack.com/feed"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "posts.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

ALLOWED = {"p","br","em","strong","i","b","u","blockquote",
           "h2","h3","h4","ul","ol","li","hr","a","img","figure","figcaption"}
VOID = {"br","hr","img","input"}
DROP_TAGS = {"form","script","style","svg","button","input","iframe"}
DROP_DIV_CLASS = ("subscription","subscribe","paywall","poll","share","button","embed")


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


class Cleaner(HTMLParser):
    """Keep an allow-list of tags; drop Substack widgets and chrome."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0

    def _start(self, tag, d):
        if tag == "a" and d.get("href"):
            return '<a href="%s" target="_blank" rel="noopener">' % escape(d["href"], quote=True)
        if tag == "img" and d.get("src"):
            return '<img src="%s" alt="%s">' % (escape(d["src"], quote=True), escape(d.get("alt",""), quote=True))
        if tag == "p" and "text-align" in d.get("style",""):
            m = re.search(r"text-align:\s*[a-z]+", d["style"])
            if m:
                return '<p style="%s">' % m.group(0)
        return "<%s>" % tag

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if self.skip:
            if tag not in VOID:
                self.skip += 1
            return
        if tag in DROP_TAGS:
            if tag not in VOID:
                self.skip += 1
            return
        if tag == "div":
            if any(k in d.get("class","") for k in DROP_DIV_CLASS):
                self.skip += 1
            return  # otherwise unwrap (keep children)
        if tag in ALLOWED:
            self.out.append(self._start(tag, d))

    def handle_startendtag(self, tag, attrs):
        if self.skip:
            return
        d = dict(attrs)
        if tag == "img" and d.get("src"):
            self.out.append(self._start("img", d))
        elif tag in ("br","hr"):
            self.out.append("<%s>" % tag)

    def handle_endtag(self, tag):
        if self.skip:
            if tag not in VOID:
                self.skip -= 1
            return
        if tag == "div" or tag in DROP_TAGS:
            return
        if tag in ALLOWED and tag not in VOID:
            self.out.append("</%s>" % tag)

    def handle_data(self, data):
        if not self.skip:
            self.out.append(escape(data, quote=False))


def _aster_to_hr(html):
    def repl(m):
        inner = re.sub(r"<[^>]+>", "", m.group(1)).replace(" ", " ").strip()
        if inner and re.fullmatch(r"[\*\s]+", inner) and "*" in inner:
            return "<hr>"
        return m.group(0)
    return re.sub(r"<p[^>]*>(.*?)</p>", repl, html, flags=re.S)


def clean(html):
    c = Cleaner()
    c.feed(html)
    out = "".join(c.out)
    out = _aster_to_hr(out)
    out = re.sub(r"<p[^>]*>(?:\s|&nbsp;)*</p>", "", out)
    out = re.sub(r"(?:<hr>\s*){2,}", "<hr>", out)
    return out.strip()


class BodyGrab(HTMLParser):
    """Capture inner HTML of the first <div class="... body markup ...">."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.cap = False
        self.depth = 0
        self.buf = []

    def _a(self, attrs):
        return "".join(' %s="%s"' % (k, (v or "")) for k, v in attrs)

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if not self.cap and tag == "div" and "body markup" in d.get("class",""):
            self.cap = True
            self.depth = 1
            return
        if self.cap:
            if tag == "div":
                self.depth += 1
            self.buf.append("<%s%s>" % (tag, self._a(attrs)))

    def handle_startendtag(self, tag, attrs):
        if self.cap:
            self.buf.append("<%s%s/>" % (tag, self._a(attrs)))

    def handle_endtag(self, tag):
        if self.cap:
            if tag == "div":
                self.depth -= 1
                if self.depth == 0:
                    self.cap = False
                    return
            self.buf.append("</%s>" % tag)

    def handle_data(self, data):
        if self.cap:
            self.buf.append(data)

    def handle_entityref(self, name):
        if self.cap:
            self.buf.append("&%s;" % name)

    def handle_charref(self, name):
        if self.cap:
            self.buf.append("&#%s;" % name)


def extract_web_body(html):
    g = BodyGrab()
    g.feed(html)
    return "".join(g.buf).strip() or None


def slug_from(link):
    m = re.search(r"/p/([^/?#]+)", link)
    return m.group(1) if m else re.sub(r"[^a-z0-9]+", "-", link.lower()).strip("-")


def main():
    feed = fetch(FEED_URL)
    posts = []
    for item in re.findall(r"<item>(.*?)</item>", feed, re.S):
        def grab(tag):
            m = re.search(r"<%s>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</%s>" % (tag, tag), item, re.S)
            return m.group(1).strip() if m else ""
        title = unescape(grab("title"))
        link = grab("link")
        try:
            date = parsedate_to_datetime(grab("pubDate")).strftime("%b %Y")
        except Exception:
            date = ""

        body_html = None
        try:
            body_html = extract_web_body(fetch(link))
        except Exception as e:
            print("warn: web fetch failed for %s (%s)" % (link, e), file=sys.stderr)
        if not body_html:  # fall back to (possibly stale) feed content
            ce = re.search(r"<content:encoded>(.*?)</content:encoded>", item, re.S)
            body_html = ce.group(1) if ce else ""
            body_html = re.sub(r"^<!\[CDATA\[", "", body_html)
            body_html = re.sub(r"\]\]>$", "", body_html)

        posts.append({
            "title": title,
            "slug": slug_from(link),
            "date": date,
            "url": link,
            "body": clean(body_html),
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print("wrote %d post(s) to %s" % (len(posts), OUT_PATH))


if __name__ == "__main__":
    main()
