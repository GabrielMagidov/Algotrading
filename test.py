import platform

is_mac = platform.system() == 'Darwin'
is_windows = platform.system() == 'Windows'

print("Is macOS:", is_mac)
print("Is Windows:", is_windows)
