import uiautomation as auto

root = auto.GetRootControl()
for win in root.GetChildren():
    name = win.Name
    cls = win.ClassName
    if any(k in name.lower() or k in cls.lower() for k in ["chrome", "gmail", "edge", "mail"]):
        print(f"Found Window: Name='{name}' | Class='{cls}'")
