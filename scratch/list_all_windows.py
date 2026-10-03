import uiautomation as auto

root = auto.GetRootControl()
for win in root.GetChildren():
    if win.Name:
        print(f"Name='{win.Name}' | Class='{win.ClassName}'")
