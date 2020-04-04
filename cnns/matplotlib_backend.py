import matplotlib

gui_env = ['Agg', 'TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
backend = None
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        backend = gui
        break
    except:
        continue
# backend = matplotlib.get_backend()
# print("Using:", backend)