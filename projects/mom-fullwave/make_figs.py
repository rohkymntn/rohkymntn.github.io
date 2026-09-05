"""Generate the figures for the MoM project write-up from real solver runs.

    python3 make_figs.py            # all figures, into this folder
    python3 make_figs.py dipole sphere   # a subset
"""
import sys, time, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, "/Users/rohk/agentic-RF")  # path to the agentic-RF checkout
OUT = str(__import__('pathlib').Path(__file__).resolve().parent)

from agentic_rf.geometry.templates import get_template
from agentic_rf.geometry.base import AntennaGeometry, TraceSegment, Point3D, Port, Box3D
from agentic_rf.utils.materials import COPPER, MUSCLE
from agentic_rf.utils.constants import C_0, ETA_0
from agentic_rf.solver.mom import (MoMSolver, extract_layout, compute_rwg, EFIEAssembler, locate_gap_port,
                                   far_field, plane_wave_excitation, bistatic_rcs, mesh_layout)
from agentic_rf.solver.mom.gmsh_mesher import mesh_sphere
from agentic_rf.solver.analytical import AnalyticalSolver
import scipy.linalg as sla

CLAY, INK, MUTED, GREEN, LINE, PAPER = "#BE5B3A", "#23211B", "#807A6B", "#4F7F5B", "#E2DCCC", "#FBF9F3"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.7,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=170); plt.close(); print("saved", name, flush=True)

def strip_mesh(L, w, nx, ny):
    from agentic_rf.solver.mom.mesh import SurfaceMesh
    xs = np.linspace(-L/2, L/2, nx+1); ys = np.linspace(-w/2, w/2, ny+1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1); tris = []
    for i in range(nx):
        for j in range(ny):
            a=i*(ny+1)+j; b=(i+1)*(ny+1)+j; c=(i+1)*(ny+1)+j+1; d=i*(ny+1)+j+1
            tris += [[a,b,c],[a,c,d]] if (i+j)%2==0 else [[a,b,d],[b,c,d]]
    return SurfaceMesh(V, np.array(tris))

def dipole_geom(L, w):
    g = AntennaGeometry(template_name="dipole")
    g.traces.append(TraceSegment(Point3D(-L/2,0,0), Point3D(L/2,0,0), width=w, thickness=35e-6))
    g.ports.append(Port(Point3D(-w,0,0), Point3D(w,0,0))); g.compute_bounding_box(); return g

def draw_mesh(ax, mesh, values=None, cmap="inferno", lw=0.25, alpha=1.0):
    V = mesh.vertices*1e3; T = mesh.triangles
    polys = V[T]
    if values is None:
        pc = Poly3DCollection(polys, facecolors="#D9CDB8", edgecolors=INK, linewidths=lw, alpha=alpha)
    else:
        norm = plt.Normalize(values.min(), values.max()); cm = plt.get_cmap(cmap)
        pc = Poly3DCollection(polys, facecolors=cm(norm(values)), edgecolors="k", linewidths=0.1)
    ax.add_collection3d(pc)
    lo, hi = V.min(0), V.max(0); c = (lo+hi)/2; r = (hi-lo).max()/2
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r*0.6, c[2]+r*0.6)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.grid(False); ax.set_facecolor(PAPER)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.set_facecolor(PAPER); pane.set_edgecolor(LINE)
    return pc

which = sys.argv[1:] or ["all"]
def want(k): return "all" in which or k in which

# ── 1. dipole validation ────────────────────────────────────────────────
if want("dipole"):
    L, w = 0.15, 0.01
    mesh = strip_mesh(L, w, 30, 2); rwg = compute_rwg(mesh); asm = EFIEAssembler(mesh, rwg, quad_points=6)
    port = locate_gap_port(mesh, rwg, np.array([0,-w/2,0]), np.array([0,w/2,0]), np.array([1.,0,0]), tol=1e-9)
    freqs = np.linspace(0.5e9, 1.5e9, 81); Z=[]; D=[]
    for f in freqs:
        k = 2*np.pi*f/C_0; Zm = asm.assemble(k, ETA_0, conductor_loss=False)
        I = np.linalg.solve(Zm, port.excitation_vector(rwg)); Z.append(port.input_impedance(rwg, I))
    Z = np.array(Z); i = np.argmin(np.abs(Z.imag[:60]))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    axs[0].plot(freqs/1e9, Z.real, color=CLAY, lw=2); axs[0].set_title("Input resistance (150 × 10 mm strip, 148 unknowns)"); axs[0].set_xlabel("GHz"); axs[0].set_ylabel("R (Ω)")
    axs[0].axvline(freqs[i]/1e9, color=MUTED, ls=":"); axs[0].annotate(f"{Z.real[i]:.1f} Ω at {freqs[i]/1e9:.3f} GHz\n(L = {L*freqs[i]/C_0:.3f} λ)", (freqs[i]/1e9, Z.real[i]), xytext=(0.55, 300), fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    axs[1].plot(freqs/1e9, Z.imag, color=INK, lw=2); axs[1].axhline(0, color=MUTED, ls=":"); axs[1].set_title("Input reactance"); axs[1].set_xlabel("GHz"); axs[1].set_ylabel("X (Ω)")
    s11 = 20*np.log10(np.abs((Z-50)/(Z+50)))
    axs[2].plot(freqs/1e9, s11, color=GREEN, lw=2); axs[2].axhline(-10, color=MUTED, ls=":"); axs[2].set_title("|S11| (50 Ω)"); axs[2].set_xlabel("GHz"); axs[2].set_ylabel("dB")
    save("fig_dipole_validation")
    # pattern at resonance
    k = 2*np.pi*freqs[i]/C_0; Zm = asm.assemble(k, ETA_0, conductor_loss=False); I = np.linalg.solve(Zm, port.excitation_vector(rwg))
    ff = far_field(asm, I, k, ETA_0, 181, 73)
    P_in = 0.5*np.real(port.voltage*np.conj(port.terminal_current(rwg, I)))
    Ddb = 10*np.log10(np.maximum(ff.directivity, 1e-6))
    fig = plt.figure(figsize=(9, 3.8))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    th = ff.theta; e_plane = Ddb[:, 0]; ax.plot(th, e_plane, color=CLAY, lw=2, label="E-plane (xz), MoM")
    ax.plot(-th, e_plane, color=CLAY, lw=2)
    ideal = 10*np.log10(1.64*(np.cos(np.pi/2*np.sin(th))/np.maximum(np.abs(np.cos(th)),1e-6))**2 + 1e-9)
    ax.plot(th, ideal, color=INK, lw=1, ls="--", label="ideal λ/2 dipole"); ax.plot(-th, ideal, color=INK, lw=1, ls="--")
    ax.set_theta_zero_location("N"); ax.set_rlim(-30, 3); ax.set_title(f"Directivity in the xz plane, peak {ff.peak_directivity_dbi:.2f} dBi", pad=14); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    Jc = asm.current_at_centroids(I); x = mesh.centroids[:,0]*1e3; o = np.argsort(x)
    ax2.plot(x[o], np.abs(Jc[o,0]), ".", color=INK, ms=4); ax2.set_xlabel("position along dipole (mm)"); ax2.set_ylabel("|Jx| (A/m per V)")
    ax2.set_title(f"Current at resonance · P_rad/P_in = {ff.P_rad/P_in:.4f}")
    save("fig_dipole_pattern")

# ── 2. sphere RCS vs Mie ────────────────────────────────────────────────
if want("sphere"):
    from scipy.special import spherical_jn, spherical_yn
    def mie(a, k, n_max=40):
        ka=k*a; tot=0j
        for n in range(1, n_max+1):
            jn=spherical_jn(n,ka); yn=spherical_yn(n,ka); h2=jn-1j*yn
            djn=spherical_jn(n,ka,derivative=True); dyn=spherical_yn(n,ka,derivative=True); dh2=djn-1j*dyn
            an=jn/h2; bn=(jn+ka*djn)/(h2+ka*dh2); tot+=(-1)**n*(2*n+1)*(an-bn)
        return (2*np.pi/k)**2/(4*np.pi)*abs(tot)**2
    a=0.1; fs=np.linspace(0.3e9, 1.4e9, 12)
    mesh = mesh_sphere(a, C_0/1.4e9/10); rwg=compute_rwg(mesh); asm=EFIEAssembler(mesh, rwg, quad_points=6)
    mom=[]; ref=[]
    for f in fs:
        k=2*np.pi*f/C_0; Z=asm.assemble(k, ETA_0, conductor_loss=False)
        V=plane_wave_excitation(asm,k,np.array([0,0,-1.]),np.array([1.,0,0])); I=np.linalg.solve(Z,V)
        mom.append(bistatic_rcs(asm,I,k,ETA_0,np.array([[0,0,1.]]))[0]); ref.append(mie(a,k))
    kas=2*np.pi*fs/C_0*a; kf=np.linspace(0.3,3.0,300); reff=[mie(a,kk/a) for kk in kf]
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    axs[0].plot(kf, 10*np.log10(np.array(reff)/(np.pi*a**2)), color=INK, lw=1.5, label="Mie series")
    axs[0].plot(kas, 10*np.log10(np.array(mom)/(np.pi*a**2)), "o", color=CLAY, ms=6, label=f"MoM, {rwg.num_basis} unknowns")
    axs[0].set_xlabel("ka"); axs[0].set_ylabel("σ / πa²  (dB)"); axs[0].set_title("PEC sphere, monostatic RCS"); axs[0].legend()
    axs[1].plot(kas, 10*np.log10(np.array(mom)/np.array(ref)), "o-", color=GREEN); axs[1].axhline(0,color=MUTED,ls=":")
    axs[1].set_ylim(-1,1); axs[1].set_xlabel("ka"); axs[1].set_ylabel("MoM − Mie (dB)"); axs[1].set_title("error")
    save("fig_sphere_mie")
    fig=plt.figure(figsize=(4.5,4)); ax=fig.add_subplot(111, projection="3d"); draw_mesh(ax, mesh); ax.set_title(f"sphere mesh, {mesh.n_triangles} triangles"); ax.view_init(20, 30); save("fig_sphere_mesh")

# ── 3. template meshes and currents ─────────────────────────────────────
if want("meshes"):
    for name, f0, f1, ftarget, view in [("mifa", 0.4e9, 1.6e9, 1.4e9, (35, -60)), ("pifa", 1.5e9, 3.0e9, 2.1e9, (30, -55)), ("patch", 3.0e9, 5.0e9, 3.9e9, (40, -60))]:
        t = get_template(name); p = t.get_default_params()
        if name != "mifa": p["substrate_er"] = 1.0
        g = t.generate_geometry(p); s = MoMSolver(); m = s.prepare(g, f1)
        res = s.simulate(g, f0, f1, n_freq=21, prepared=m)
        fig = plt.figure(figsize=(11, 4.2))
        ax = fig.add_subplot(1, 2, 1, projection="3d"); draw_mesh(ax, m.mesh); ax.view_init(*view)
        ax.set_title(f"{name.upper()} mesh · {m.mesh.n_triangles} triangles · {m.rwg.num_basis} RWG")
        i = int(np.argmin(np.abs(res.frequencies_hz - ftarget)))
        J = res.surface_current_mag[i]; Jdb = np.maximum(20*np.log10(J/J.max()+1e-12), -40)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d"); pc = draw_mesh(ax2, m.mesh, Jdb); ax2.view_init(*view)
        ax2.set_title(f"|J| at {res.frequencies_hz[i]/1e9:.2f} GHz (dB rel. max)")
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-40, 0), cmap="inferno"), ax=ax2, shrink=0.6, pad=0.1, label="dB")
        save(f"fig_mesh_current_{name}")
        json.dump({"f": (res.frequencies_hz/1e6).tolist(), "R": res.z_input[:,0].real.tolist(), "X": res.z_input[:,0].imag.tolist(),
                   "s11": res.s11.tolist(), "eff": res.radiation_efficiency.tolist(), "gain": res.peak_gain_dbi.tolist(),
                   "n": m.rwg.num_basis, "nt": m.mesh.n_triangles, "time": res.solve_time_seconds}, open(f"{OUT}/sweep_{name}.json", "w"))

# ── 4. debugging: LDL vs LU ─────────────────────────────────────────────
if want("ldl"):
    t = get_template("pifa"); p = t.get_default_params(); p["substrate_er"] = 1.0; g = t.generate_geometry(p)
    s = MoMSolver(); m = s.prepare(g, 2.6e9); asm = m.assembler; port = m.ports[0]
    freqs = np.linspace(1.8e9, 2.6e9, 33); Zs = {"sym": [], "gen": []}
    for f in freqs:
        k = 2*np.pi*f/C_0; Z = asm.assemble(k, ETA_0, omega=2*np.pi*f, conductor_loss=True)
        V = port.excitation_vector(m.rwg)
        for mode in ("sym", "gen"):
            I = sla.solve(Z if mode == "sym" else 0.5*(Z+Z.T), V, assume_a=mode); Zs[mode].append(port.input_impedance(m.rwg, I))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, lab in ((axs[0], "real", "R (Ω)"), (axs[1], "imag", "X (Ω)")):
        ax.plot(freqs/1e9, getattr(np.array(Zs["sym"]), key), color=MUTED, lw=1.5, label="LDLᵀ on raw Z (assume_a='sym')")
        ax.plot(freqs/1e9, getattr(np.array(Zs["gen"]), key), color=CLAY, lw=2.2, label="symmetrized Z, full LU")
        ax.set_xlabel("GHz"); ax.set_ylabel(lab)
    axs[0].legend(fontsize=8); axs[0].set_title("Air PIFA input impedance: same matrix, two solvers")
    sym_err = np.max(np.abs(Z-Z.T))/np.max(np.abs(Z)); axs[1].set_title(f"‖Z−Zᵀ‖/‖Z‖ = {sym_err:.1e}, cond(Z) ≈ 10⁵")
    save("fig_debug_ldl")

# ── 5. debugging: end caps ──────────────────────────────────────────────
if want("caps"):
    from agentic_rf.solver.mom import layout as lay_mod
    L, w = 0.15, 0.01; g = dipole_geom(L, w); freqs = np.linspace(0.7e9, 1.1e9, 25)
    out = {}
    for cap in (True, False):
        orig = lay_mod._ribbon_polygon
        lay_mod._ribbon_polygon = (lambda pts, width, closed, cap_start=True, cap_end=True, _o=orig: _o(pts, width, closed, cap, cap))
        try:
            s = MoMSolver(conductor_loss=False); res = s.simulate(g, freqs[0], freqs[-1], n_freq=len(freqs))
        finally:
            lay_mod._ribbon_polygon = orig
        out[cap] = res
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for cap, col, lab in ((True, MUTED, "ribbon with w/2 end caps (160 mm)"), (False, CLAY, "flush ribbon (150 mm)")):
        ax.plot(out[cap].frequencies_hz/1e9, out[cap].z_input[:,0].imag, color=col, lw=2, label=lab)
    ax.axhline(0, color=MUTED, ls=":"); ax.set_xlabel("GHz"); ax.set_ylabel("X (Ω)"); ax.legend(fontsize=8)
    ax.set_title("Strip dipole reactance: the 7 % that was not physics")
    save("fig_debug_caps")

# ── 6. debugging: degenerate mesh ───────────────────────────────────────
if want("degenerate"):
    from agentic_rf.solver.mom import gmsh_mesher as gm
    t = get_template("mifa"); g = t.generate_geometry(t.get_default_params())
    s = MoMSolver(mesh_refinement=0.6); lay = extract_layout(g); s._assign_mesh_sizes(lay, 1.5e9)
    good, _ = gm.mesh_layout(lay)
    # reproduce the failure: re-enable the redundant embed step and disable the sanity gate
    src = open(gm.__file__).read()
    real_sane = gm._mesh_is_sane; gm._mesh_is_sane = lambda mesh: (True, "")
    real_tf = gm._transfinite_ribbons
    def tf_with_embed(gmsh, panels, children):
        gm._embed_junction_curves(gmsh, panels, children); real_tf(gmsh, panels, children)
    gm._transfinite_ribbons = tf_with_embed
    try:
        bad, _ = gm.mesh_layout(lay)
    finally:
        gm._mesh_is_sane = real_sane; gm._transfinite_ribbons = real_tf
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for j, (mesh, title) in enumerate(((bad, "over-constrained (embedded curve + fragment line)"), (good, "fixed (explicit fragment lines only)"))):
        V = mesh.tri_vertices; Lg = np.stack([np.linalg.norm(V[:,(i+1)%3]-V[:,i],axis=1) for i in range(3)],1)
        q = 4*np.sqrt(3)*mesh.areas/np.sum(Lg**2,1)
        cm = plt.get_cmap("viridis"); ax_ = axs[j]
        for tidx in np.where(np.abs(mesh.normals[:, 2]) > 0.9)[0]:
            P = V[tidx][:, [0, 1]] * 1e3
            ax_.fill(P[:, 0], P[:, 1], color=cm(q[tidx]), ec="k", lw=0.3)
        ax_.set_aspect("equal"); ax_.grid(False)
        ax_.set_xlim(-20.4, -17.6); ax_.set_ylim(-0.6, 0.6); ax_.set_xlabel("x (mm)"); ax_.set_ylabel("y (mm)")
        ax_.set_title(f"{title}\nmin triangle quality {q.min():.3f}", fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"), ax=axs, shrink=0.7, label="triangle quality 4√3·A/Σl²")
    fig.text(0.42, 0.02, "top view of the meander trace between the short pin (x = −20 mm) and the feed pin (x = −18 mm)", ha="center", fontsize=9, color=MUTED)
    plt.savefig(f"{OUT}/fig_debug_degenerate.png", dpi=170); plt.close(); print("saved fig_debug_degenerate", flush=True)

# ── 7. mesh convergence ─────────────────────────────────────────────────
if want("convergence"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name, f0, f1, er in ((axs[0], "pifa", 1.8e9, 2.4e9, 1.0), (axs[1], "mifa", 1.1e9, 1.5e9, None)):
        t = get_template(name); p = t.get_default_params()
        if er: p["substrate_er"] = er
        g = t.generate_geometry(p)
        for ref, col in ((1.0, MUTED), (0.7, GREEN), (0.5, CLAY)):
            s = MoMSolver(mesh_refinement=ref); m = s.prepare(g, f1); res = s.simulate(g, f0, f1, n_freq=13, prepared=m)
            ax.plot(res.frequencies_hz/1e9, np.abs(res.z_input[:,0]), color=col, lw=2, label=f"refinement {ref}: N = {m.rwg.num_basis}, {res.solve_time_seconds:.0f} s")
        ax.set_yscale("log"); ax.set_xlabel("GHz"); ax.set_ylabel("|Z_in| (Ω)"); ax.set_title(f"{name.upper()} mesh convergence"); ax.legend(fontsize=8)
    save("fig_convergence")

# ── 8. fidelity: analytical vs MoM ──────────────────────────────────────
if want("fidelity"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name, f0, f1, er in ((axs[0], "pifa", 1.0e9, 3.0e9, 1.0), (axs[1], "patch", 3.0e9, 5.0e9, 1.0)):
        t = get_template(name); p = t.get_default_params(); p["substrate_er"] = er; g = t.generate_geometry(p)
        an = AnalyticalSolver(t).simulate(g, f0, f1, 201); mom = MoMSolver().simulate(g, f0, f1, n_freq=41)
        ax.plot(an.frequencies_hz/1e9, an.s11, color=MUTED, ls="--", lw=1.5, label=f"closed-form model (f_res {t.estimate_resonance_hz(p)/1e9:.2f} GHz)")
        ax.plot(mom.frequencies_hz/1e9, mom.s11, color=CLAY, lw=2.2, label=f"full-wave MoM ({mom.diagnostics.n_basis} unknowns)")
        ax.axhline(-10, color=MUTED, ls=":"); ax.set_xlabel("GHz"); ax.set_ylabel("|S11| (dB)"); ax.set_ylim(-25, 1)
        ax.set_title(f"{name.upper()} (air substrate)"); ax.legend(fontsize=8, loc="lower left")
    save("fig_fidelity_s11")

# ── 9. implant: free space vs muscle ────────────────────────────────────
if want("implant"):
    g = dipole_geom(0.03, 0.002); freqs = (300e6, 3.0e9)
    fs = MoMSolver(conductor_loss=False).simulate(g, *freqs, n_freq=41)
    mu = MoMSolver(medium=MUSCLE, conductor_loss=False).simulate(g, *freqs, n_freq=41)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, res, title in ((axs[0], fs, "free space"), (axs[1], mu, "immersed in muscle (εr 57, σ 0.8 S/m)")):
        z = res.z_input[:, 0]; f = res.frequencies_hz/1e9
        ax.plot(f, z.real, color=CLAY, lw=2, label="R"); ax.plot(f, z.imag, color=INK, lw=2, label="X")
        ax.axhline(0, color=MUTED, ls=":"); ax.set_xlabel("GHz"); ax.set_ylabel("Ω"); ax.legend(fontsize=8)
        if title == "free space":
            note = "\nλ/2 resonance at ≈ 4.6 GHz, capacitive over the whole band"
        else:
            i = int(np.argmax(z.imag)); note = f"\nnear-resonant at {f[i]:.2f} GHz: X = {z.imag[i]:+.1f} Ω, R = {z.real[i]:.0f} Ω, heavily damped"
        ax.set_title(f"30 × 2 mm dipole, {title}" + note, fontsize=10)
    axs[0].set_ylim(-1500, 200)
    save("fig_implant_medium")

# ── 10. loop inductance vs theory ───────────────────────────────────────
if want("loop"):
    radii = [5, 8, 12, 15, 20]; L_mom = []; L_th = []
    for r in radii:
        t = get_template("loop"); p = t.get_default_params(); p["radius_mm"] = r; p["trace_width_mm"] = 1.0; g = t.generate_geometry(p)
        f = 100e6; res = MoMSolver(conductor_loss=False).simulate(g, f, f*1.5, n_freq=2)
        L_mom.append(res.z_input[0,0].imag/(2*np.pi*f)*1e9)
        a = 1e-3/4; R = r*1e-3; L_th.append(4e-7*np.pi*R*(np.log(8*R/a)-2)*1e9)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(radii, L_th, "--", color=INK, label="μ₀R[ln(8R/a)−2], a = w/4")
    ax.plot(radii, L_mom, "o", color=CLAY, ms=7, label="MoM, X/ω at 100 MHz")
    ax.set_xlabel("loop radius (mm)"); ax.set_ylabel("inductance (nH)"); ax.set_title("Single-turn loop, 1 mm strip"); ax.legend(fontsize=8)
    save("fig_loop_inductance")
    json.dump({"radii": radii, "L_mom": L_mom, "L_th": L_th}, open(f"{OUT}/loop.json", "w"))

# ── 11. performance scaling ─────────────────────────────────────────────
if want("perf"):
    Ns=[]; t_static=[]; t_fill=[]; t_solve=[]
    for ref in (1.6, 1.2, 0.9, 0.7, 0.55):
        t = get_template("patch"); p = t.get_default_params(); p["substrate_er"]=1.0; g = t.generate_geometry(p)
        s = MoMSolver(mesh_refinement=ref); m = s.prepare(g, 4e9); asm = m.assembler
        k=2*np.pi*4e9/C_0; t0=time.time(); Z=asm.assemble(k, ETA_0, omega=2*np.pi*4e9); tf=time.time()-t0
        t0=time.time(); sla.solve(0.5*(Z+Z.T), m.ports[0].excitation_vector(m.rwg), assume_a="gen"); ts=time.time()-t0
        Ns.append(m.rwg.num_basis); t_static.append(asm.stats.static_build_seconds); t_fill.append(tf); t_solve.append(ts)
        print("perf", ref, Ns[-1], tf, ts, flush=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.loglog(Ns, t_static, "s-", color=GREEN, label="static part (once per mesh)")
    ax.loglog(Ns, t_fill, "o-", color=CLAY, label="dynamic fill (per frequency)")
    ax.loglog(Ns, t_solve, "^-", color=INK, label="LU solve (per frequency)")
    n=np.array(Ns); ax.loglog(n, t_fill[1]*(n/n[1])**2, ":", color=MUTED, label="∝ N²"); ax.loglog(n, t_solve[1]*(n/n[1])**3, "--", color=MUTED, label="∝ N³")
    ax.set_xlabel("RWG unknowns N"); ax.set_ylabel("seconds"); ax.set_title("Cost scaling (air patch, laptop CPU)"); ax.legend(fontsize=7)
    save("fig_performance")
    json.dump({"N": Ns, "static": t_static, "fill": t_fill, "solve": t_solve}, open(f"{OUT}/perf.json", "w"))

# ── 12. analytic singular integral ──────────────────────────────────────
if want("singular"):
    from agentic_rf.solver.mom.analytic import static_potential_integrals
    from agentic_rf.solver.mom.quadrature import triangle_rule
    tri = np.array([[0,0,0],[1,0,0],[0,1,0]], float)
    xs = np.linspace(-0.6, 1.6, 221); ys = np.linspace(-0.6, 1.6, 221); X, Y = np.meshgrid(xs, ys)
    obs = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    I0, _ = static_potential_integrals(obs, np.repeat(tri[None], len(obs), 0))
    bary, w = triangle_rule(6); pts = bary @ tri
    I0q = np.array([0.5*np.sum(w/np.maximum(np.linalg.norm(pts - o, axis=1), 1e-12)) for o in obs])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, val, title in ((axs[0], I0, "closed form ∫_T dS′/R"), (axs[1], I0q, "6-point sampled Σ wᵢ/Rᵢ")):
        im = ax.imshow(np.clip(val.reshape(X.shape), 0, 4), extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", cmap="magma", vmin=0, vmax=4)
        ax.plot([0,1,0,0],[0,0,1,0], color="w", lw=1); ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="potential (source triangle of unit legs)")
    plt.savefig(f"{OUT}/fig_singular_integral.png", dpi=170); plt.close(); print("saved fig_singular_integral", flush=True)
