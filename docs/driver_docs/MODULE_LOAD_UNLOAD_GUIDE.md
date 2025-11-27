# NVIDIA Kernel Module Load/Unload Guide

This guide provides scripts and instructions for unloading and loading NVIDIA kernel modules, both from the system-installed location and from locally compiled builds.

## Prerequisites

- Root/sudo access
- Display manager may need to be stopped (for graphical systems)
- No critical GPU-dependent applications running

## Important Notes

- **SSH is safe**: Unloading/reloading modules will NOT affect SSH connections
- **Display will restart**: Graphical desktop sessions will be logged out when display manager is stopped
- **Temporary vs Permanent**: Using `insmod` loads modules temporarily; they revert to system defaults on reboot
- **System modules**: Located in `/lib/modules/$(uname -r)/updates/dkms/`
- **Local modules**: Located in `kernel-open/*.ko` after building

## Module Dependencies

Modules must be loaded/unloaded in the correct dependency order:

**Unload order (reverse dependency):**
1. nvidia_uvm
2. nvidia_drm
3. nvidia_modeset
4. nvidia (base module)

**Load order:**
1. nvidia (base module)
2. nvidia_modeset
3. nvidia_drm
4. nvidia_uvm

## Script 1: Unload All NVIDIA Modules

```bash
#!/bin/bash
# unload_nvidia_modules.sh
# Unloads all NVIDIA kernel modules

echo "=== Stopping nvidia-persistenced service ==="
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo killall -9 nvidia-persistenced 2>/dev/null || true

echo "=== Stopping display manager ==="
# This will log out graphical session
sudo systemctl stop gdm3 2>/dev/null || \
sudo systemctl stop gdm 2>/dev/null || \
sudo systemctl stop lightdm 2>/dev/null || true

# Give services time to stop
sleep 2

echo "=== Unloading NVIDIA modules in dependency order ==="
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo rmmod nvidia_drm 2>/dev/null || true
sudo rmmod nvidia_modeset 2>/dev/null || true
sudo rmmod nvidia 2>/dev/null || true

# Verify all unloaded
if lsmod | grep -q nvidia; then
    echo "ERROR: Some NVIDIA modules still loaded:"
    lsmod | grep nvidia
    exit 1
fi

echo "=== All NVIDIA modules unloaded successfully ==="
```

## Script 2: Load System Modules (Using modprobe)

```bash
#!/bin/bash
# load_system_modules.sh
# Loads NVIDIA modules from system installation (/lib/modules/)

echo "=== Loading NVIDIA modules from system ==="

# Load modules in correct dependency order
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

echo "=== Verifying modules loaded ==="
lsmod | grep nvidia

echo ""
echo "=== Module information ==="
modinfo nvidia | grep -E "^(filename|version|srcversion):"
modinfo nvidia_uvm | grep -E "^(filename|version|srcversion):"

echo ""
echo "=== Checking kernel messages ==="
sudo dmesg | tail -20 | grep -i nvidia

echo ""
echo "=== Restarting display manager ==="
sudo systemctl start gdm3 2>/dev/null || \
sudo systemctl start gdm 2>/dev/null || \
sudo systemctl start lightdm 2>/dev/null || true

echo ""
echo "=== Done! System modules loaded ==="
```

## Script 3: Load Local Modules (Using insmod)

```bash
#!/bin/bash
# load_local_modules.sh
# Loads NVIDIA modules from local kernel-open/ directory
# These modules are NOT installed to the system and will revert on reboot

# Check if we're in the correct directory
if [ ! -f "kernel-open/nvidia.ko" ]; then
    echo "ERROR: kernel-open/nvidia.ko not found!"
    echo "Please run this script from the open-gpu-kernel-modules root directory"
    exit 1
fi

echo "=== Loading NVIDIA modules from kernel-open/ ==="

# Load modules in correct dependency order
sudo insmod kernel-open/nvidia.ko
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load nvidia.ko"
    exit 1
fi

sudo insmod kernel-open/nvidia-modeset.ko
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load nvidia-modeset.ko"
    sudo rmmod nvidia
    exit 1
fi

sudo insmod kernel-open/nvidia-drm.ko
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load nvidia-drm.ko"
    sudo rmmod nvidia_modeset nvidia
    exit 1
fi

sudo insmod kernel-open/nvidia-uvm.ko
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load nvidia-uvm.ko"
    sudo rmmod nvidia_drm nvidia_modeset nvidia
    exit 1
fi

echo "=== Verifying modules loaded ==="
lsmod | grep nvidia

echo ""
echo "=== Module information ==="
modinfo nvidia | grep -E "^(filename|version):"
modinfo nvidia_uvm | grep -E "^(filename|version):"

echo ""
echo "=== Checking kernel messages for custom build messages ==="
sudo dmesg | tail -30 | grep -E "(Custom build|yunwei37|NVIDIA)"

echo ""
echo "=== Restarting display manager ==="
sudo systemctl start gdm3 2>/dev/null || \
sudo systemctl start gdm 2>/dev/null || \
sudo systemctl start lightdm 2>/dev/null || true

echo ""
echo "=== Done! Local modules loaded ==="
echo "NOTE: These modules will revert to system defaults on reboot"
```

## Script 4: Complete Reload Workflow

```bash
#!/bin/bash
# reload_nvidia_modules.sh
# Complete workflow: unload, then load local modules

set -e

echo "========================================="
echo "NVIDIA Module Reload Script"
echo "========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "kernel-open/nvidia.ko" ]; then
    echo "ERROR: kernel-open/nvidia.ko not found!"
    echo "Please run this script from the open-gpu-kernel-modules root directory"
    exit 1
fi

# Step 1: Unload
echo "Step 1: Unloading existing modules..."
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo systemctl stop gdm3 2>/dev/null || sudo systemctl stop gdm 2>/dev/null || sudo systemctl stop lightdm 2>/dev/null || true
sleep 2

sudo rmmod nvidia_uvm 2>/dev/null || true
sudo rmmod nvidia_drm 2>/dev/null || true
sudo rmmod nvidia_modeset 2>/dev/null || true
sudo rmmod nvidia 2>/dev/null || true

if lsmod | grep -q nvidia; then
    echo "ERROR: Some NVIDIA modules still loaded"
    lsmod | grep nvidia
    exit 1
fi
echo "✓ All modules unloaded"
echo ""

# Step 2: Clear dmesg for clean output (optional)
echo "Step 2: Clearing dmesg buffer..."
sudo dmesg -c > /dev/null
echo "✓ dmesg cleared"
echo ""

# Step 3: Load local modules
echo "Step 3: Loading local modules..."
sudo insmod kernel-open/nvidia.ko
sudo insmod kernel-open/nvidia-modeset.ko
sudo insmod kernel-open/nvidia-drm.ko
sudo insmod kernel-open/nvidia-uvm.ko
echo "✓ All modules loaded"
echo ""

# Step 4: Verify
echo "Step 4: Verification..."
echo "Loaded modules:"
lsmod | grep nvidia
echo ""
echo "Custom build messages:"
sudo dmesg | grep "Custom build by yunwei37"
echo ""

# Step 5: Restart display
echo "Step 5: Restarting display manager..."
sudo systemctl start gdm3 2>/dev/null || sudo systemctl start gdm 2>/dev/null || sudo systemctl start lightdm 2>/dev/null || true
echo "✓ Display manager restarted"
echo ""

echo "========================================="
echo "Module reload complete!"
echo "========================================="
```

## One-Liner Commands

### Quick unload all modules:
```bash
sudo systemctl stop nvidia-persistenced && sudo systemctl stop gdm3 && sleep 2 && sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
```

### Quick load system modules:
```bash
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm && sudo systemctl start gdm3
```

### Quick load local modules:
```bash
sudo insmod kernel-open/nvidia.ko && sudo insmod kernel-open/nvidia-modeset.ko && sudo insmod kernel-open/nvidia-drm.ko && sudo insmod kernel-open/nvidia-uvm.ko && sudo systemctl start gdm3
```

## Troubleshooting

### Module won't unload (busy)
```bash
# Check what's using the module
lsof /dev/nvidia*

# Check module usage count
lsmod | grep nvidia

# Force stop all GPU processes (careful!)
sudo fuser -k /dev/nvidia*
```

### Wrong module loaded
```bash
# Check which file the loaded module came from
modinfo nvidia | grep filename

# System module will show: /lib/modules/.../nvidia.ko.zst
# Local module will also show system path in modinfo (kernel limitation)
# But you can verify by checking build date in dmesg
```

### Verify custom build is loaded
```bash
# Check for custom messages in kernel log
sudo dmesg | grep "Custom build by yunwei37"

# Check build timestamp
sudo dmesg | grep "NVRM: loading" | grep "yunwei37@lab"
```

### Module version mismatch
```bash
# Check kernel version
uname -r

# Check module vermagic
modinfo nvidia | grep vermagic

# Rebuild if necessary
make clean && make -j$(nproc) modules
```

## Permanent Installation (Not Recommended for Testing)

If you want to permanently install your custom modules (will persist across reboots):

```bash
# Backup original modules first!
sudo cp /lib/modules/$(uname -r)/updates/dkms/nvidia*.ko.zst ~/nvidia_backup/

# Install new modules
sudo make modules_install

# Update module dependencies
sudo depmod -a

# Reboot to use new modules
sudo reboot
```

**Warning**: Only do this if you're sure your custom build is stable!

## Reverting to Original Modules

If you installed custom modules and want to revert:

```bash
# Reinstall original NVIDIA driver
sudo apt-get install --reinstall nvidia-driver-575

# Or restore from backup
sudo cp ~/nvidia_backup/nvidia*.ko.zst /lib/modules/$(uname -r)/updates/dkms/

# Update dependencies
sudo depmod -a

# Reboot
sudo reboot
```

## References

- NVIDIA driver source: `/usr/src/nvidia-575.57.08/`
- Installed modules: `/lib/modules/$(uname -r)/updates/dkms/`
- Local build: `kernel-open/*.ko`
- Kernel messages: `sudo dmesg | grep -i nvidia`
- Module info: `modinfo nvidia`, `modinfo nvidia_uvm`
