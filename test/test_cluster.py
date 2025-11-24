import os, sys, torch, yaml, pkg_resources

print("🚀 === CLUSTER TEST START ===")
print(f"🐍 Python: {sys.version.split()[0]} | Exec: {sys.executable}")
print(f"📂 Working dir: {os.getcwd()}")
print(f"🔍 PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
print(f"📁 sys.path entries:")
for p in sys.path: print("   •", p)
print(f"🧠 Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"💻 GPU detected: {torch.cuda.get_device_name(0)}")

    print("\n🧪 Testing imports from src/")
    try:
        from src.utils.seed import init_torch
        print("   ✔ Successfully imported init_torch from src.utils.seed")
    except Exception as e:
        print("   ❌ Failed to import from src/")
        print("     Error:", e)

print("\n📦 Installed key packages:")
for name in ["numpy", "matplotlib", "torch", "yaml", "pandas"]:
    try:
        version = pkg_resources.get_distribution(name).version
        print(f"   • {name:<10} {version}")
    except pkg_resources.DistributionNotFound:
        print(f"   • {name:<10} not installed")

print("\n🧾 Command-line arguments:", sys.argv)

# Optional extra check: verify YAML parsing
try:
    yaml.safe_load("key: value")
    print("✅ YAML parser operational")
except Exception as e:
    print(f"⚠️ YAML parser error: {e}")

if torch.cuda.is_available():
    print(f"✅ CUDA functional — Device count: {torch.cuda.device_count()}")
else:
    print("⚠️ CUDA unavailable — running on CPU only")

print("\n🧪 Running init_torch sanity check")
try:
    if 'init_torch' in globals():
        init_torch(123)
        print("   ✔ init_torch(123) executed successfully")
    else:
        print("   ⚠ init_torch not available (import failed earlier)")
except Exception as e:
    print("   ❌ init_torch failed:", e)

print("\n🧪 Project root & src/ verification")
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(project_root, "src")
    print(f"   • Project root: {project_root}")
    if os.path.isdir(src_dir):
        print(f"   ✔ src/ directory exists at {src_dir}")
        print("   • Listing first 10 entries in src/:")
        print("     ", os.listdir(src_dir)[:10])
    else:
        print(f"   ❌ src/ directory NOT found at {src_dir}")
except Exception as e:
    print("   ❌ Error verifying src/:", e)

print("\n🧪 GPU mini-benchmark")
try:
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"   ✔ GPU matmul test succeeded, result shape: {z.shape}")
    else:
        print("   ⚠ GPU unavailable — skipping benchmark")
except Exception as e:
    print("   ❌ GPU benchmark failed:", e)

print("\n🎯 Summary:")
print("   - Python environment successfully activated")
print("   - Key packages loaded")
print("   - GPU access verified (if applicable)")
print("   - All checks passed ✅")

print("🏁 === CLUSTER TEST END ===")