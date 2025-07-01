import os
import site
import sys

def get_package_sizes(site_packages_path):
    package_sizes = []

    for item in os.listdir(site_packages_path):
        item_path = os.path.join(site_packages_path, item)
        if os.path.isdir(item_path):
            total_size = 0
            for root, dirs, files in os.walk(item_path):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp):
                        total_size += os.path.getsize(fp)
            package_sizes.append((item, total_size))

    package_sizes.sort(key=lambda x: x[1], reverse=True)
    return package_sizes

# 현재 사용 중인 site-packages 경로
site_packages_path = site.getsitepackages()[0] if hasattr(site, 'getsitepackages') else site.getusersitepackages()

print(f"\n🔍 Site-packages 위치: {site_packages_path}\n")
package_sizes = get_package_sizes(site_packages_path)

print("📦 용량 큰 패키지 Top 20:\n")
for name, size in package_sizes[:20]:
    print(f"{name:<40} {size / (1024 * 1024):.2f} MB")
