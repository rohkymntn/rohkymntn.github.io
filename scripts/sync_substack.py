#!/usr/bin/env python3
"""Fetch a Substack RSS feed and write posts.json in the site's structure.

Stdlib only (works locally and in CI without pip installs).
Usage: python3 scripts/sync_substack.py [feed_url] [out_path]
"""
import sys, re, json, urllib.request
from html import escape, unescape
from html.parser import HTMLParser
from email.utils import parsedate_to_datetime

FEED_URL = sys.argv[1] if len(sys.argv) > 1 else "https://kyleroh.substack.com/feed"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "posts.json"

ALLOWED = {"p","br","em","strong","i","b","u","blockquote",
           "h2","h3","h4","ul","ol","li","hr","a","img","figure","figcaption"}
VOID = {"br","hr","img","input"}
DROP_TAGS = {"form","script","style","svg","button","input","iframe"}
DROP_DIV_CLASS = ("subscription","subscribe","paywall","poll","share","button","embed")


class Cleaner(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0  # depth of subtree we're discarding

    def _start(self, tag, d, selfclose=False):
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
            cls = d.get("class","")
            if any(k in cls for k in DROP_DIV_CLASS):
                self.skip += 1
            return  # otherwise unwrap (keep children, drop the div tag)
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
        if self.skip:
            return
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
    out = re.sub(r"<p[^>]*>(?:\s|&nbsp;)*</p>", "", out)  # drop empty paragraphs
    out = re.sub(r"(?:<hr>\s*){2,}", "<hr>", out)          # collapse repeated rules
    return out.strip()


def slug_from(link):
    m = re.search(r"/p/([^/?#]+)", link)
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9]+", "-", link.lower()).strip("-")


def main():
    req = urllib.request.Request(FEED_URL, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        xml = r.read().decode("utf-8", "replace")

    posts = []
    for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
        def grab(tag):
            m = re.search(r"<%s>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</%s>" % (tag, tag), item, re.S)
            return m.group(1).strip() if m else ""
        title = unescape(grab("title"))
        link = grab("link")
        ce = re.search(r"<content:encoded>(.*?)</content:encoded>", item, re.S)
        body_raw = ce.group(1) if ce else ""
        body_raw = re.sub(r"^<!\[CDATA\[", "", body_raw)
        body_raw = re.sub(r"\]\]>$", "", body_raw)
        try:
            dt = parsedate_to_datetime(grab("pubDate"))
            date = dt.strftime("%b %Y")
        except Exception:
            date = ""
        posts.append({
            "title": title,
            "slug": slug_from(link),
            "date": date,
            "url": link,
            "body": clean(body_raw),
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print("wrote %d post(s) to %s" % (len(posts), OUT_PATH))


if __name__ == "__main__":
    main()
