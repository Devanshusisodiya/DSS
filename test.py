from tkinter.ttk import *
from tkinter import *
from time import *
import threading as t

def main():
    root = Tk()
    root.geometry("400x400")

    pb = Progressbar(
        root,
        orient='horizontal',
        mode='indeterminate',
        length=100
    )

    pb.place(relx=0.5, rely=0.5)

    def delay():
        from scipy.optimize import dual_annealing
        import numpy as np
        import pandas as pd
        # getting some data
        data = pd.read_csv("C://dataset12.csv")
        t = pd.Series(range(1, len(data.Time) + 1))
        mt = data.CDF

        # defining the objective function

        def GO(a, b, t):
            return a*(1-np.exp(-b*t))

        def obj(x, t, mt):
            res = mt - GO(x[0], x[1], t)
            squared = np.mean(res**2)
            return squared
        
        bounds = [
            (1000, 1.5e7),
            (0.1e-4, 0.4e-1)
        ]

        ret = dual_annealing(obj, args=(t, mt), bounds=bounds, maxiter=1000)
        print(ret)

    t.Thread(target=delay).start()
    pb.start()
    pb.stop()
    # button = Button(root, text='start', command=pb.start)
    # button.place(relx=0.5, rely=0.7)

    root.mainloop()


main()