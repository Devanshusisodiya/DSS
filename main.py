from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from tkinter.messagebox import *
from routines import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#------------------------------------------DATASET LOADER AND VIEWER--------------------------------------------
# SUBROUTINE TO LOAD DATA
def loadData(path, stringvar):
    try:
        data = pd.read_csv(path)
        stringvar.set(path)
        showinfo(title='Load successful', message='The dataset has been loaded successfuly.')
    except FileNotFoundError or PermissionError:
        return showerror(title="Incorrect path", message='The path entered does not exist. Please make sure to enter correct path.')

# WINDOW TO VIEW DATA
def viewData(path):
    #checking if the path entered or not
    try:
        data = pd.read_csv(f'{path}')   

        rootView = Toplevel(root)
        rootView.title('Dataset')
        rootView.geometry('620x250')
        rootView.resizable(False, False)

        # define columns
        columns = ('month', 'faults', 'cdf')
        tree = Treeview(rootView, columns=columns, show='headings')
        # define headings
        tree.heading('month', text='Month')
        tree.heading('faults', text='Faults')
        tree.heading('cdf', text='Cummulative Faults')
        # add data to the treeview
        for i in range(len(data)):
            tree.insert('', END, values=tuple(data.loc[i]))

        # add a scrollbar
        scrollbar = Scrollbar(rootView, orient= VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        tree.grid(row=0, column=0)
        scrollbar.grid(row=0, column=1, sticky='ns')

        rootView.mainloop()
    except FileNotFoundError:
        showerror(title="Dataset not loaded", message='Please enter the correct path of dataset and press the load button')

#-----------------------------------------------------------------------------------------------------------
#--------------MODEL SELECTOR, PARAMETER ESTIMATION AND RANK CALCULATION------------------------------------

# I DONT KNOW WHY I NAMED IT COMPUTE BUT ITS JUST THE MODEL PARAMETER ESTIMATOR
def compute(path):
    data = pd.read_csv(path)

    X = data['Time']
    Y = data['CDF']

    rootComp = Toplevel(root)
    rootComp.geometry("200x300")
    rootComp.title("Models")
    
    # I THINK I SHOULD ALSO INCLUDE THE PARAMETER ESTIMATION INSIDE THESE FUNCTIONS ONLY
    # WITHIN THE _sub METHOD, AND MAINTAIN THE DATABASE(GLOBAL MODEL DICTIONARY WITH THE ESTIMATED PARAMETERS)
    # IT IS GOING TO CONTAIN MULTIPLE FUNCTIONS IN ORDER TO INCLUDE MORE MODELS IN THE SYSTEM
    
    # BELOW IS FOR MODEL 1 ---> GO MODEL. ILL UPDATE THE NAMES LATER ON WHEN THE SYSTEM WILL BE WORKING FLAWLESSLY
    # NAMES HAVE BEEN CHANGED BECAUSE THE SYSTEM HAD NO TROUBLE INCORPORATING ANOTHER MODEL
    def goModelParamEst(modelObject):
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get())
            ]
            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
            
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")
        param2Label = Label(rootPar, text="b")
        param1 = Entry(rootPar) # THIS IS PARAMETER 'a' FOR GO MODEL
        param2 = Entry(rootPar) # THIS IS PARAMETER 'b' FOR GO MODEL
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        submitAndEstimate.grid(row=2, column=1)
        removeModel.grid(row=3, column=1)

        rootPar.mainloop()

    # THIS IS FOR MODEL 2 ---> DELAYED S
    def delayedSParamEst(modelObject):
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get())
            ]
            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")
        param2Label = Label(rootPar, text="b")
        param1 = Entry(rootPar) # THIS IS PARAMETER 'a' FOR DELAYED S MODEL
        param2 = Entry(rootPar) # THIS IS PARAMETER 'b' FOR DELAYED S MODEL
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        submitAndEstimate.grid(row=2, column=1)
        removeModel.grid(row=3, column=1)

        rootPar.mainloop()

    # THIS IS FOR MODEL 3 ---> INFLECTION S
    def inflectionSParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")  # THIS IS PARAMETER 'a' FOR INFLECTION S MODEL
        param2Label = Label(rootPar, text="b")  # THIS IS PARAMETER 'b' FOR INFLECTION S MODEL
        param3Label = Label(rootPar, text="\u03B2") #beta
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 4 ---> YAMADA RAYLEIGH
    def yamadaRayleighParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B2") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B3") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        param4 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        param4Label.grid(row=3, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        param4.grid(row=3, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 5 ---> YAMADA IMPERFECT 1
    def yamadaImperfect1ParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA IMPERFECT 1 MODEL
        param2Label = Label(rootPar, text="b")      # THIS IS PARAMETER 'b' FOR YAMADA IMPERFECT 1 MODEL
        param3Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA IMPERFECT 1 MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 6 ---> YAMADA IMPERFECT 2
    def yamadaImperfect2ParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA IMPERFECT 1 MODEL
        param2Label = Label(rootPar, text="b")      # THIS IS PARAMETER 'b' FOR YAMADA IMPERFECT 1 MODEL
        param3Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA IMPERFECT 1 MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 7 ---> YAMADA EXPONENTIAL
    def yamadaExponentialParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B2") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B3") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        param4 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        param4Label.grid(row=3, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        param4.grid(row=3, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 8 ---> VTUB SHAPED MODEL
    def vtubParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get()),
                float(param5.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param5label = Label(rootPar, text="n")      # THIS IS PARAMETER 'n' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        param4 = Entry(rootPar)
        param5 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        param4Label.grid(row=3, column=0)
        param5label.grid(row=4, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        param4.grid(row=3, column=2)
        param5.grid(row=4, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 9 ---> RMD MODEL
    def rmdParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        param4 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        param4Label.grid(row=3, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        param4.grid(row=3, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()
    
    # THIS IS FOR MODEL 10 ---> CHANGS MODEL
    def changParamEst(modelObject):    
        rootPar = Toplevel(rootComp)

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get()),
                float(param5.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            modelMap[modelObject.name] = list(minimizationResults.x)
            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param5label = Label(rootPar, text="n")      # THIS IS PARAMETER 'n' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar)
        param2 = Entry(rootPar)
        param3 = Entry(rootPar)
        param4 = Entry(rootPar)
        param5 = Entry(rootPar)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.grid(row=0, column=0)
        param2Label.grid(row=1, column=0)
        param3Label.grid(row=2, column=0)
        param4Label.grid(row=3, column=0)
        param5label.grid(row=4, column=0)
        # ENTRY PLACING
        param1.grid(row=0, column=2)
        param2.grid(row=1, column=2)
        param3.grid(row=2, column=2)
        param4.grid(row=3, column=2)
        param5.grid(row=4, column=2)
        submitAndEstimate.grid(row=5, column=2)
        removeModel.grid(row=6, column=2)

        rootPar.mainloop()

    # ADDING BUTTONS TO INPUT MODEL PARAMETERS
    model1 = Button(rootComp, text='GO Model', command=lambda: goModelParamEst(gomodel))
    model2 = Button(rootComp, text="Yamada Rayleigh", command=lambda: yamadaRayleighParamEst(yamadaR))
    model3 = Button(rootComp, text="Delayed S", command=lambda: delayedSParamEst(delayedS))
    model4 = Button(rootComp, text="Inflection S", command=lambda: inflectionSParamEst(inflectionS))
    model5 = Button(rootComp, text="Yamada Imperfect 1", command=lambda: yamadaImperfect1ParamEst(yamadaImperfect1))
    model6 = Button(rootComp, text="Yamada Imperfect 2", command=lambda: yamadaImperfect2ParamEst(yamadaImperfect2))
    model7 = Button(rootComp, text="Yamada Exponential", command=lambda: yamadaExponentialParamEst(yamadaExponential))
    model8 = Button(rootComp, text="Vtub Shaped", command=lambda: vtubParamEst(vtub))
    model9 = Button(rootComp, text="RMD", command=lambda: rmdParamEst(rmd))
    model10 = Button(rootComp, text="Chang et al\'s", command=lambda: changParamEst(changs))

    # ADDING PREVIOUSLY CREATED BUTTONS
    model1.pack()
    model2.pack()
    model3.pack()
    model4.pack()
    model5.pack()
    model6.pack()
    model7.pack()
    model8.pack()
    model9.pack()
    model10.pack()

    rootComp.mainloop()

# THIS ROUTINE IS TO DISPLAY ESTIMATED PARAMETERS FOR THE MODELS
def display():
    # CHECK FOR NONE NULL MODELS, AND PRINT RESPECTIVE VALUES
    computedModels = []
    for model in modelMap:
        param = modelMap[model]
        if param:
            computedModels.append(model)
    if len(computedModels) == 0:
        showinfo(title='Parameter Display', message='No parameters have been estimated')
    else:
        rootDisp = Toplevel(root)
        rootDisp.title("Estimated Parameters")

    
        for i in range(len(computedModels)):
            model = computedModels[i]
            labelParams = paramMap[model]
            estimatedParams = modelMap[model]
            paramString = ""
            # CREATING THE STRING FOR PARAMETERS
            for j in range(len(labelParams)):
                # PARAMETER STRING = PARAMETER LABEL + PARAMETER VALUE
                parameter = labelParams[j]
                parameterValue = "{:.4e}".format(estimatedParams[j])
                paramString += parameter + " : " + parameterValue + "\n"
            # A LITTLE FORMATTING OF DATA
            model = model + "  "
            paramString = "\n" + paramString
            # ADDING MODEL AND PARAMETERS FOR DISPLAY IN THE WINDOW
            modelLabel = Label(rootDisp, text=model)
            modelParams = Label(rootDisp, text=paramString)
            modelLabel.grid(row=i, column=0)
            modelParams.grid(row=i, column=1)

       

        rootDisp.mainloop()
    pass

# THIS ROUTINE HERE IS TO CALCULATE RANK, BUT JUST A LOT MORE EFFICIENT IN PERFORMANCE AND BETTER LOOKING
def computeR(path):

    data = []
    # CHECKING IF MODEL PARAMETERS ARE EVEN ENTERED BY THE USER OR NOT, IF THEY ARE THE DATA LIST IS POPULATED
    # WITH MODEL ALONG WITH THE ESTIMATED PARAMETERS
    for model in modelMap:
        param = modelMap[model]
        print(model, "--->", param)
        if param:
            data.append([model, param]) # THE FIRST ARG ---> STRING, SECOND ARG ---> PARAMETERS

    if len(data) <= 1:
        showerror(title='Cannot run TOPSIS', message="None or less models have been selected, TOPSIS requires atleast 2 models.")
    else:
        # GETTING THE DATA FROM DATASET IN ORDER TO CALCULATE PERFORMANCE CRITERIA
        dataset = pd.read_csv(path)
        X = dataset['Time']
        Y = dataset['CDF']

        # FOR CREATING ENTRIES IN data:
        rootRank = Toplevel(root)
        rootRank.title("Results")
        rootRank.resizable(False, False)

        modelRoutines = [
            gomodel,
            yamadaR,
            delayedS,
            inflectionS,
            yamadaImperfect1,
            yamadaImperfect2,
            yamadaExponential,
            vtub,
            rmd,
            changs
        ]
        criteriaData = []


        # VALIDATING THE DATABASE KEYS WITH MODEL NAMES 
        for selected in data:
            criterias = []
            for model in modelRoutines:
                if selected[0] == model.name:
                    params = selected[1]
                    mse = model.mse(params, X, Y)
                    mae = model.mae(params, X, Y)
                    r2 = model.rsquare(params, X, Y)
                    adr = model.adrsquare(params, X, Y)
                    pp = model.PP(params, X, Y)             # IF ANY PROBLEM OCCURS IT IS IN PP, AIC, MEOP or TS  
                    aic = model.aic(params, X, Y)
                    meop = model.meop(params, X, Y)
                    theil = model.theil(params, X, Y) 

                    criterias = [mse, mae, r2, adr, aic, pp, meop, theil] # WHILE ADDING COLUMNS TO THE TREEVIEW MAKE SURE TO ENTER THE COLUMNS IN THIS ORDER OF CRITERIA ONLY
                    break
            criterias.insert(0, selected[0])
            criteriaData.append(criterias)

        # print(criteriaData)
        # CALCULATE TOPSIS OVER HERE--------------------------------


        cdata = []
        for modelData in criteriaData:
            cdata.append(modelData[1:])
        cdata = np.array(cdata)

        n = cdata.shape[0] # number of models
        m = cdata.shape[1] # number of criterias

        # SO FAR SO GOOD
        # CALCULATING THE WEIGHT MATRIX ----------------------------------------------

        # normalizing the weight matrix
        P = cdata / np.sum(cdata, axis=0)
        # calculating the entropy vector
        e = (-1) * np.sum(P * np.log(P), axis=0) / np.log(n)
        # calculating the degree of diversification
        d = 1-e
        # calculating the weights
        w = d / np.sum(d)

        # CALCULATED THE WEIGHT MATRIX -----------------------------------------------
        # CALCULATING THE RANK -------------------------------------------------------
        y = cdata / np.sqrt(np.sum(cdata**2, axis=0))
        v = w * y

        vpos = []
        vneg = []

        criteria_map = {
            'mse': v[:, 0],
            'mae': v[:, 1],
            'r2': v[:, 2],
            'adjr2': v[:, 3],
            'aic': v[: 4],
            'pp': v[:, 5],
            'meop': v[:, 6],
            'theil': v[: 7]
        }
        maximizer = ['r2', 'adjr2']
        minimizer = ['mse', 'mae', 'pp', 'meop', 'aic', 'theil']

        for criteria in criteria_map:
            if criteria in maximizer:
                best = np.amax(criteria_map[criteria])
                worst = np.amin(criteria_map[criteria])
                vpos.append(best)
                vneg.append(worst)
            if criteria in minimizer:
                best = np.amin(criteria_map[criteria])
                worst = np.amax(criteria_map[criteria])
                vpos.append(best)
                vneg.append(worst)

        # converting ideal best and ideal worst data to usable form
        vpos = np.array(vpos)
        vneg = np.array(vneg)

        spos = np.sqrt(np.sum( (v-vpos)**2 , axis=1))
        sneg = np.sqrt(np.sum( (v-vneg)**2 , axis=1))

        # final relative closeness results
        c = sneg / (spos + sneg)

        # ranking the models
        ranked = {}
        rankArr = np.array([0 for _ in range(n)])
        initialRank = 1

        for _ in range(n):
            # populating rank list
            index = np.argmin(c)
            ranked[index] = initialRank
            # discarding the considered index
            c[index] = np.inf
            initialRank += 1

        for i in range(n):
            rankArr[i] = ranked[i]

        rankArr = np.array([[i] for i in rankArr])
        # CALCULATED THE RANKS -----------------------------------------------

        modelsWithRank = []
        criteriasWithRank = np.append(cdata, rankArr, 1)

        for i in range(len(criteriasWithRank)):
            row = list(criteriasWithRank[i])
            row.insert(0, data[i][0])
            modelsWithRank.append(row)

        print(modelsWithRank)
        #-----------------------------------------------------------
        # ADDING TO TABLE
        columns = ('models', 'mse', 'mae', 'rsquare', 'adrsquare', 'aic', 'pp', 'meop', 'theil', 'rank')
        columnsText = ('Models', 'MSE', 'MAE', 'R Squared', 'Adj. R Squared', 'AIC', 'PP', 'MEOP', 'TS', 'Rank')
        tree = Treeview(rootRank, columns=columns, show='headings')

        for i in range(len(columns)):
            tree.heading(columns[i], text=columnsText[i])
            tree.column(columns[i], minwidth=110, width=100)
        

        for i in modelsWithRank:
            # SHORTENING THE NUMBERS TILL 5 DECIMAL PLACES
            model = i[0]
            temp = [np.round_(j, decimals=5) for j in i[1:]]
            temp.insert(0, model)
            # ADDING TO THE TREE VIEW
            tree.insert('', END, values=tuple(temp))

        tree.grid(row=0, column=0)
        rootRank.mainloop()

# THIS ROUTINE ACTS AS A DISPATCHER TO ABOVE ROUTINES ---> display(), compute(), computeR()
def models(path):
    try:
        data = pd.read_csv(path)
        
        global rootModels
        rootModels = Toplevel(root)

        # MODEL AND ESTIMATED PARAMETER HASHMAP
        global modelMap
        global paramMap
        modelMap = {
            'GO': None,
            'Delayed S Shaped': None,
            'Inflection S Shaped': None,
            'Yamada Rayleigh': None,
            'Yamada Imperfect 1': None,
            'Yamada Imperfect 2': None,
            'Yamada Exponential': None,
            'Vtub Shaped': None,
            'RMD': None,
            'Chang et al\'s': None
        }
        paramMap = {
            'GO': ['a', 'b'],                                          # a, b
            'Delayed S Shaped': ['a', 'b'],                            # a, b
            'Inflection S Shaped': ['a', 'b', '\u03B2'],               # a, b, beta
            'Yamada Rayleigh': ['a', '\u03B1', '\u03B2', '\u03B3'],    # a, alpha, beta, gamma
            'Yamada Imperfect 1': ['a', 'b', '\u03B1'],                # a, b, alpha
            'Yamada Imperfect 2': ['a', 'b', '\u03B1'],                # a, b, alpha
            'Yamada Exponential': ['a', '\u03B1', '\u03B2', '\u03B3'], # a, alpha, beta, gamma
            'Vtub Shaped': ['a', 'b', '\u03B1', '\u03B2', 'n'],        # a, b, alpha, beta, n 
            'RMD': ['a', 'b', '\u03B1', '\u03B2'],                     # a, b, alpha, beta 
            'Chang et al\'s':['a', 'b', '\u03B1', '\u03B2', 'n']       # a, b, alpha, beta, n
        }

        comp = Button(rootModels, text='Compute', command=lambda: compute(path))
        disp = Button(rootModels, text='Display', command=lambda: display())
        compR = Button(rootModels, text='Compute Rank', command=lambda: computeR(path))
        comp.pack()
        disp.pack()
        compR.pack()
        rootModels.mainloop()
    except FileNotFoundError:
        showerror(title="Dataset not loaded", message='Please enter the correct path of dataset and press the load button')

#------------------------------------------------------------------------------------------------------------------------

# MAIN FUNCTION
def main():
    global root
    root = Tk()

    path = StringVar()

    modelButton = Button(root, text='View Models', command=lambda: models(pathEntry.get())) # SUBSTITUTE WITH THE models OVER HERE
    pathEntry = Entry(root)
    loadButton = Button(root, text='Load Dataset', command=lambda: loadData(pathEntry.get(), path))
    viewButton = Button(root, text='View Dataset', command=lambda: viewData(pathEntry.get()))

    # PLACING THE BUTTONS  
    modelButton.grid(row=1, column=0)
    pathEntry.grid(row=0, column=2)
    loadButton.grid(row=1, column=2)
    viewButton.grid(row=2, column=2)

    root.mainloop()

main()