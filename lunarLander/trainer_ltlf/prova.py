import lydia
import pkgutil

print("Percorso di installazione:")
print(lydia.__path__)

print("\nSottomoduli disponibili:")
for item in pkgutil.iter_modules(lydia.__path__):
    print(item.name)
