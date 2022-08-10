import re
from tkinter.ttk import *
from tkinter import *

def main():
    root = Tk()
    root.geometry("300x220")

    def getWeights(weightOp):
        if weightOp.get() == 1:
            return
        elif weightOp.get() == 2:
            return
        elif weightOp.get() == 3:
            return

    def getRanks(rankOp, weightVec):
        if rankOp.get() == 1:
            return
        elif rankOp.get() == 2:
            return
    
    def rankCalc(weightOp, rankOp):
        weights = getWeights(weightOp)
        ranks = getRanks(rankOp, weights)

        rootRank = Tk()
        rootRank.geometry('200x200')

        rootRank.mainloop()

    weightOp = IntVar()
    rankOp = IntVar()

    w1 = Radiobutton(root, text='AHP', value=1, variable=weightOp)
    w2 = Radiobutton(root, text='EWM', value=2, variable=weightOp)
    r1 = Radiobutton(root, text='CODAS', value=1, variable=rankOp)
    r2 = Radiobutton(root, text='TOPSIS', value=2, variable=rankOp)
    submit = Button(root, text='Submit', command=lambda: rankCalc(weightOp, rankOp))

    w1.grid(row=0, column=0)
    w2.grid(row=1, column=0)
    r1.grid(row=0, column=2)
    r2.grid(row=1, column=2)
    submit.grid(row=2, column=1)

    root.mainloop()


main()