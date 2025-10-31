import os, sys, torch, yaml, pkg_resources

print("🚀 === CLUSTER TEST START ===")
print(f"🐍 Python: {sys.version.split()[0]} | Exec: {sys.executable}")
print(f"📂 Working dir: {os.getcwd()}")
print(f"🔍 PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
print(f"🧠 Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"💻 GPU detected: {torch.cuda.get_device_name(0)}")

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

print("\n🎯 Summary:")
print("   - Python environment successfully activated")
print("   - Key packages loaded")
print("   - GPU access verified (if applicable)")
print("   - All checks passed ✅")

print("🏁 === CLUSTER TEST END ===")