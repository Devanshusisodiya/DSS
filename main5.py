from re import L
from tkinter.messagebox import *
from tkinter import *
from tkinter.ttk import *
import numpy as np
import pandas as pd
import tkinter as tk
from scipy.optimize import dual_annealing
from routines import *

# routine to load the dataset
def loadDataset(master):
    pass

# routine to estimate model parameters
def paramEstimation():
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
            estimatedMSE = mseMap[model]
            paramString = ""
            # CREATING THE STRING FOR PARAMETERS
            for j in range(len(labelParams)):
                # PARAMETER STRING = PARAMETER LABEL + PARAMETER VALUE
                parameter = labelParams[j]
                parameterValue = "{:.4e}".format(estimatedParams[j])
                paramString += parameter + "   :   " + parameterValue + "\n"
            # A LITTLE FORMATTING OF DATA
            model = model + "  "
            # ADDING MSE TO ESTIMATED PARAMETERS
            paramString += "MSE" + "   :   " + str(np.round_(estimatedMSE, decimals=4)) + "\n"
            paramString = "\n" + paramString
            # ADDING MODEL AND PARAMETERS FOR DISPLAY IN THE WINDOW
            modelLabel = Label(rootDisp, text=model)
            modelParams = Label(rootDisp, text=paramString)
            modelLabel.grid(row=i, column=0, padx=25)
            modelParams.grid(row=i, column=1, padx=25)

        rootDisp.mainloop()

# routine to calculate ranks
def rankModels(genGraph = False):
    if genGraph:
        pass
    else:
        pass

    pass

# routine to calculate reliability
def reliabilityCalc():
    pass

# defining main routine
def main():
    global root
    root = Tk()
    root.title("Sim. Annealing DSS")
    root.geometry("700x500")
    
    # some variables that are required
    path = StringVar()
    loaded = BooleanVar()
    
    # hashmaps for essential data
    global paramMap, modelMap, mseMap
    modelMap = {
        'GO model': None,
        'Delayed S-Shaped model': None,
        'Inflection S-Shaped model': None,
        'PNZ model': None,
        'Yamada Imperfect 1 model': None,
        'Vtub-Shaped model': None,
        'Yamada Imperfect 2 model': None,
        'RMD model`': None,
        'Yamada Exponential model': None,
        'Chang et al\'s model': None
    }
    mseMap = {
        'GO model': np.inf,
        'Delayed S-Shaped model': np.inf,
        'Inflection S-Shaped model': np.inf,
        'PNZ model': np.inf,
        'Yamada Imperfect 1 model': np.inf,
        'Yamada Imperfect 2 model': np.inf,
        'Yamada Exponential model': np.inf,
        'Vtub-Shaped model': np.inf,
        'RMD model': np.inf,
        'Chang et al\'s model': np.inf
    }
    paramMap = {
        'GO model': ['a', 'b'],                                          # a, b
        'Delayed S-Shaped model': ['a', 'b'],                            # a, b
        'Inflection S-Shaped model': ['a', 'b', '\u03B2'],               # a, b, beta
        'PNZ model': ['a', 'b', '\u03B1', '\u03B2'],                     # a, b, alpha, beta
        'Yamada Imperfect 1 model': ['a', 'b', '\u03B1'],                # a, b, alpha
        'Yamada Imperfect 2 model': ['a', 'b', '\u03B1'],                # a, b, alpha
        'Yamada Exponential model': ['a', '\u03B1', '\u03B2', '\u03B3'], # a, alpha, beta, gamma
        'Vtub-Shaped model': ['a', 'b', '\u03B1', '\u03B2', 'N'],        # a, b, alpha, beta, N 
        'RMD model': ['a', 'b', '\u03B1', '\u03B2'],                     # a, b, alpha, beta 
        'Chang et al\'s model':['a', 'b', '\u03B1', '\u03B2', 'N']       # a, b, alpha, beta, N
    }
    
    # creating tab controller
    tabController = Notebook(root)

    # defining the tabs
    global datasetTab, paramTab, rankingTab, reliabilityTab
    datasetTab = Frame(tabController)
    paramTab = Frame(tabController)
    rankingTab = Frame(tabController)
    reliabilityTab = Frame(tabController)

    # tabs of DSS
    # using a map to get the tab names and the tabs themselves
    tabs = {
        datasetTab: "Data Set Handling",
        paramTab: "Parameter Estimation & Model Selection",
        rankingTab: "Model Ranking",
        reliabilityTab: "Model Reliability"
    }
    # adding the tabs
    for i in tabs:
        tabController.add(i, text=tabs[i])
    
    # subroutine to view the dataset
    def viewDataset(stringvar):
        data = pd.read_csv(stringvar.get())
        # defining the treeview
        tree = Treeview(datasetTab, show='headings', height=9, selectmode="extended")
        # define columns
        tree['columns'] = ('time', 'cdf')
        tree.column('time', anchor=CENTER, width=350)
        tree.column('cdf', anchor=CENTER, width=350)
        # define headings
        tree.heading('time', text='Time', anchor=CENTER)
        tree.heading('cdf', text='CDF', anchor=CENTER)
        # add data to the treeview
        for i in range(len(data)):
            tree.insert('', END, values=(data.loc[i].Time, data.loc[i].CDF))
        
        # placing the tree on the canvas and removing the 
        # surrogate dataset canvas
        tree.place(relx=0, rely=0.55)
        datasetCanvas.destroy()
    
    # subroutine to load the dataset
    def loadDataset(path, stringvar, loaded):
        try:
            data = pd.read_csv(path)
            X = np.arange(1, len(data.Time)+1)
            Y = data['CDF']
            stringvar.set(path)
            loaded.set(True)
            viewDataset(stringvar)
            showinfo(title='Load successful', message='The dataset has been loaded successfuly.')
            return X, Y
        except FileNotFoundError or PermissionError:
            return showerror(title="Incorrect path", message='The path entered does not exist. Please make sure to enter correct path.')

    # dataset tab elements
    pathEntryLabel = Label(datasetTab, text="Enter path of dataset")
    pathEntry = Entry(datasetTab, width=72)
    loadButton = Button(datasetTab, text='Load Dataset', command=lambda: loadDataset(pathEntry.get(), path, loaded))

    datasetCanvas = Canvas(datasetTab, width=700, height=225, bg='#ffdac9')
    datasetMessage = tk.Label(datasetCanvas, text='Dataset Not Loaded.', bg='#ffdac9').place(relx=0.5, rely=0.5, anchor=CENTER)
    


    # placing dataset elements
    pathEntryLabel.place(relx=0.1, rely=0.1)
    pathEntry.place(relx=0.3, rely=0.1)
    loadButton.place(relx=0.3, rely=0.3, relwidth=0.25, relheight=0.1)
    # placing the dataset window
    datasetCanvas.place(relx=0, rely=0.55)
    tabController.pack(expand=1, fill='both')

    # parameter estimation and model selection
    style = Style()


    (
        style.configure('GO.TButton', background = 'red'),
        style.configure('Delayed.TButton', background = 'red'),
        style.configure('Inflection.TButton', background = 'red'),
        style.configure('YamadaRay.TButton', background = 'red'),
        style.configure('YamadaIm1.TButton', background = 'red'),
        style.configure('YamadaIm2.TButton', background = 'red'),
        style.configure('YamadaExpo.TButton', background = 'red'),
        style.configure('Vtub.TButton', background = 'red'),
        style.configure('RMD.TButton', background = 'red'),
        style.configure('Changs.TButton', background = 'red')
    )
    
    def goModelParamEst(modelObject, path, loaded):

        if loaded.get():
            # getting the data
            data = pd.read_csv(path.get())
            X = np.arange(1, len(data.Time)+1)
            Y = data.CDF
            # rest compute
            rootPar = Toplevel(paramTab)
            rootPar.title("GO Model Initial Parameters")
            rootPar.geometry("550x275")

            def _sub():
                
                bounds = [
                    (float(param1lb.get()), float(param1ub.get())),
                    (float(param2lb.get()), float(param2ub.get()))
                ]
            
                minimizationResults = dual_annealing(func=modelObject.OLS, args=(X, Y), bounds=bounds, maxiter=5000)

                estimatesdParams = list(minimizationResults.x)
                mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
                # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE PREVIOUS MSE
                if modelObject.estAgain:

                    paramStringNew = ''
                    paramStringOld = ''
                    oldParams = modelMap[modelObject.name]
                    subParamMap = paramMap[modelObject.name]
                    numParams = len(bounds)
                    
                    for i in range(numParams):
                        parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                        parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                        paramStringNew += parNew
                        paramStringOld += parOld
                    
                    res = askyesno(
                        title='MSE changed',
                        message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                            paramStringOld[:-2],
                            paramStringNew[:-2],
                            mseMap[modelObject.name],
                            mse
                        )
                    )

                    if res == True:
                        mseMap[modelObject.name] = mse
                        modelMap[modelObject.name] = list(minimizationResults.x)
                        rootPar.destroy()
                    if res == False:
                        rootPar.destroy()
                else:
                    
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    modelObject.estAgain = True
                    rootPar.destroy()
                
                    # CHANGING BUTTON COLOR
                    style.configure('GO.TButton', background='green')
                    #
                    rootPar.destroy()
                
            def _removeModel():
                try:
                    mseMap[modelObject.name] = np.inf
                    modelMap[modelObject.name] = None
                    modelObject.estAgain = False
                    # CHANGING BUTTON COLOR
                    style.configure('GO.TButton', background='red')
                    #
                    rootPar.destroy()
                    showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    showinfo(title='Model Removal', message="Error occured while removing model")

            param1Label = Label(rootPar, text="a (cummulative faults)")
            param2Label = Label(rootPar, text="b (detection rate)")
            param1lb = Entry(rootPar, width=15) # THIS IS PARAMETER 'a' FOR GO MODEL
            param2lb = Entry(rootPar, width=15) # THIS IS PARAMETER 'b' FOR GO MODEL
            param1ub = Entry(rootPar, width=15) # THIS IS PARAMETER 'a' FOR GO MODEL
            param2ub = Entry(rootPar, width=15) # THIS IS PARAMETER 'b' FOR GO MODEL
            submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
            removeModel = Button(rootPar, text="Remove", command=_removeModel)
            # LABELS PLACING
            param1Label.place(relx=0, rely=0)
            param2Label.place(relx=0, rely=0.15)
            # ENTRY PLACING
            param1lb.place(relx=0.4, rely=0)
            param2lb.place(relx=0.4, rely=0.15)
            param1ub.place(relx=0.6, rely=0)
            param2ub.place(relx=0.6, rely=0.15)
            submitAndEstimate.place(relx=0.3, rely=0.5, relwidth=0.35, relheight=0.2)
            removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)

            rootPar.mainloop()
        else:
            showinfo(title='Dataset not loaded yet', message='Please load the dataset before model selection')
    
    # THIS IS FOR MODEL 2 ---> DELAYED S
    def delayedSParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("Delayed S Shaped Model Initial Parameters")
        rootPar.geometry("550x175")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get())
            ]
            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('Delayed.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                modelObject.estAgain = False
                # CHANGING BUTTON COLOR
                style.configure('Delayed.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")
        param2Label = Label(rootPar, text="b (detection rate)")
        param1lb = Entry(rootPar, width=15) # THIS IS PARAMETER 'a' FOR GO MODEL
        param2lb = Entry(rootPar, width=15) # THIS IS PARAMETER 'b' FOR GO MODEL
        param1ub = Entry(rootPar, width=15) # THIS IS PARAMETER 'a' FOR GO MODEL
        param2ub = Entry(rootPar, width=15) # THIS IS PARAMETER 'b' FOR GO MODEL
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.15)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.15)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.15)
        submitAndEstimate.place(relx=0.3, rely=0.5, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 3 ---> INFLECTION S
    def inflectionSParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("Inflection S Shaped Initial Parameters")
        rootPar.geometry("550x200")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('Inflection.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('Inflection.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")  # THIS IS PARAMETER 'a' FOR INFLECTION S MODEL
        param2Label = Label(rootPar, text="b (detection rate)")  # THIS IS PARAMETER 'b' FOR INFLECTION S MODEL
        param3Label = Label(rootPar, text="\u03B2 (scale parameter)") #beta
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.125)
        param3Label.place(relx=0, rely=0.25)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.125)
        param3lb.place(relx=0.4, rely=0.25)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.125)
        param3ub.place(relx=0.6, rely=0.25)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 4 ---> PHAM NORDMANN ZHANG
    def pnzParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("PNZ Model Initial Parameters")
        rootPar.geometry("550x250")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('YamadaRay.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('YamadaRay.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b (detection rate)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param4lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        param4ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.1)
        param3Label.place(relx=0, rely=0.2)
        param4Label.place(relx=0, rely=0.3)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.1)
        param3lb.place(relx=0.4, rely=0.2)
        param4lb.place(relx=0.4, rely=0.3)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.1)
        param3ub.place(relx=0.6, rely=0.2)
        param4ub.place(relx=0.6, rely=0.3)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 5 ---> YAMADA IMPERFECT 1
    def yamadaImperfect1ParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("Yamada Imperfect1 Model Initial Parameters")
        rootPar.geometry("550x200")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('YamadaIm1.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('YamadaIm1.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA IMPERFECT 1 MODEL
        param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA IMPERFECT 1 MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA IMPERFECT 1 MODEL
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        

        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.125)
        param3Label.place(relx=0, rely=0.25)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.125)
        param3lb.place(relx=0.4, rely=0.25)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.125)
        param3ub.place(relx=0.6, rely=0.25)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 6 ---> YAMADA IMPERFECT 2
    def yamadaImperfect2ParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("Yamada Imperfect2 Model Initial Parameters")
        rootPar.geometry("550x200")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('YamadaIm2.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('YamadaIm2.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA IMPERFECT 1 MODEL
        param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA IMPERFECT 1 MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA IMPERFECT 1 MODEL
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        

        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.125)
        param3Label.place(relx=0, rely=0.25)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.125)
        param3lb.place(relx=0.4, rely=0.25)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.125)
        param3ub.place(relx=0.6, rely=0.25)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 7 ---> YAMADA EXPONENTIAL
    def yamadaExponentialParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("Yamada Exponential Model Initial Parameters")
        rootPar.geometry("550x250")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('YamadaExpo.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('YamadaExpo.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B3 (confidence level)") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param4lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        param4ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.1)
        param3Label.place(relx=0, rely=0.2)
        param4Label.place(relx=0, rely=0.3)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.1)
        param3lb.place(relx=0.4, rely=0.2)
        param4lb.place(relx=0.4, rely=0.3)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.1)
        param3ub.place(relx=0.6, rely=0.2)
        param4ub.place(relx=0.6, rely=0.3)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 8 ---> VTUB SHAPED MODEL
    def vtubParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("VTub Model Initial Parameters")
        rootPar.geometry("550x275")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get()),
                float(param5.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('Vtub.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('Vtub.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param5label = Label(rootPar, text="n")      # THIS IS PARAMETER 'n' FOR YAMADA RAYLEIGH MODEL
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param4lb = Entry(rootPar, width=15)
        param5lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        param4ub = Entry(rootPar, width=15)
        param5ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.09)
        param3Label.place(relx=0, rely=0.18)
        param4Label.place(relx=0, rely=0.27)
        param5label.place(relx=0, rely=0.36)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.09)
        param3lb.place(relx=0.4, rely=0.18)
        param4lb.place(relx=0.4, rely=0.27)
        param5lb.place(relx=0.4, rely=0.36)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.09)
        param3ub.place(relx=0.6, rely=0.18)
        param4ub.place(relx=0.6, rely=0.27)
        param5ub.place(relx=0.6, rely=0.36)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # THIS IS FOR MODEL 9 ---> RMD MODEL
    def rmdParamEst(modelObject):
        rootPar = Toplevel(paramTab)
        rootPar.title("RMD Model Initial Parameters")
        rootPar.geometry("550x250")

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get()),
                float(param4.get())
            ]

            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
            if modelObject.estAgain:

                paramStringNew = ''
                paramStringOld = ''
                oldParams = modelMap[modelObject.name]
                subParamMap = paramMap[modelObject.name]
                numParams = len(arrToOptimize)
                
                for i in range(numParams):
                    parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                    parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                    paramStringNew += parNew
                    paramStringOld += parOld
                
                res = askyesno(
                    title='MSE changed',
                    message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                        paramStringOld[:-2],
                        paramStringNew[:-2],
                        mseMap[modelObject.name],
                        mse
                    )
                )

                if res == True:
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    rootPar.destroy()
                if res == False:
                    rootPar.destroy()
            else:
                
                mseMap[modelObject.name] = mse
                modelMap[modelObject.name] = list(minimizationResults.x)
                modelObject.estAgain = True
                rootPar.destroy()

            # CHANGING BUTTON COLOR
            style.configure('RMD.TButton', background='green')
            #
            rootPar.destroy()
        
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                # CHANGING BUTTON COLOR
                style.configure('RMD.TButton', background='red')
                #
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param1lb = Entry(rootPar, width=15)
        param2lb = Entry(rootPar, width=15)
        param3lb = Entry(rootPar, width=15)
        param4lb = Entry(rootPar, width=15)
        param1ub = Entry(rootPar, width=15)
        param2ub = Entry(rootPar, width=15)
        param3ub = Entry(rootPar, width=15)
        param4ub = Entry(rootPar, width=15)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.1)
        param3Label.place(relx=0, rely=0.2)
        param4Label.place(relx=0, rely=0.3)
        # ENTRY PLACING
        param1lb.place(relx=0.4, rely=0)
        param2lb.place(relx=0.4, rely=0.1)
        param3lb.place(relx=0.4, rely=0.2)
        param4lb.place(relx=0.4, rely=0.3)
        param1ub.place(relx=0.6, rely=0)
        param2ub.place(relx=0.6, rely=0.1)
        param3ub.place(relx=0.6, rely=0.2)
        param4ub.place(relx=0.6, rely=0.3)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()
    
    # THIS IS FOR MODEL 10 ---> CHANGS MODEL
    def changParamEst(modelObject):
            rootPar = Toplevel(paramTab)
            rootPar.title("Chang et al's Model Initial Parameters")
            rootPar.geometry("550x275")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get()),
                    float(param4.get()),
                    float(param5.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                estimatesdParams = list(minimizationResults.x)
                mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
                # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
                if modelObject.estAgain:

                    paramStringNew = ''
                    paramStringOld = ''
                    oldParams = modelMap[modelObject.name]
                    subParamMap = paramMap[modelObject.name]
                    numParams = len(arrToOptimize)
                    
                    for i in range(numParams):
                        parNew = subParamMap[i] + "=" + "{:.4e}".format(estimatesdParams[i]) + ", "
                        parOld = subParamMap[i] + "=" + "{:.4e}".format(oldParams[i]) + ", "
                        paramStringNew += parNew
                        paramStringOld += parOld
                    
                    res = askyesno(
                        title='MSE changed',
                        message='Old parameters are {}, new parameters are {}. Previous MSE is {}, new MSE is {}. Do you want to continue?'.format(
                            paramStringOld[:-2],
                            paramStringNew[:-2],
                            mseMap[modelObject.name],
                            mse
                        )
                    )

                    if res == True:
                        mseMap[modelObject.name] = mse
                        modelMap[modelObject.name] = list(minimizationResults.x)
                        rootPar.destroy()
                    if res == False:
                        rootPar.destroy()
                else:
                    
                    mseMap[modelObject.name] = mse
                    modelMap[modelObject.name] = list(minimizationResults.x)
                    modelObject.estAgain = True
                    rootPar.destroy()

                # CHANGING BUTTON COLOR
                style.configure('Changs.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    mseMap[modelObject.name] = np.inf
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('Changs.TButton', background='red')
                    #
                    rootPar.destroy()
                    showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    showinfo(title='Model Removal', message="Error occured while removing model")

            param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
            param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
            param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
            param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
            param5label = Label(rootPar, text="n")      # THIS IS PARAMETER 'n' FOR YAMADA RAYLEIGH MODEL
            param1lb = Entry(rootPar, width=15)
            param2lb = Entry(rootPar, width=15)
            param3lb = Entry(rootPar, width=15)
            param4lb = Entry(rootPar, width=15)
            param5lb = Entry(rootPar, width=15)
            param1ub = Entry(rootPar, width=15)
            param2ub = Entry(rootPar, width=15)
            param3ub = Entry(rootPar, width=15)
            param4ub = Entry(rootPar, width=15)
            param5ub = Entry(rootPar, width=15)
            submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
            removeModel = Button(rootPar, text="Remove", command=_removeModel)
            
            # LABELS PLACING
            param1Label.place(relx=0, rely=0)
            param2Label.place(relx=0, rely=0.09)
            param3Label.place(relx=0, rely=0.18)
            param4Label.place(relx=0, rely=0.27)
            param5label.place(relx=0, rely=0.36)
            # ENTRY PLACING
            param1lb.place(relx=0.4, rely=0)
            param2lb.place(relx=0.4, rely=0.09)
            param3lb.place(relx=0.4, rely=0.18)
            param4lb.place(relx=0.4, rely=0.27)
            param5lb.place(relx=0.4, rely=0.36)
            param1ub.place(relx=0.6, rely=0)
            param2ub.place(relx=0.6, rely=0.09)
            param3ub.place(relx=0.6, rely=0.18)
            param4ub.place(relx=0.6, rely=0.27)
            param5ub.place(relx=0.6, rely=0.36)
            submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
            removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)

            rootPar.mainloop()

    # defining model buttons
    
    model1 = Button(paramTab, text='GO Model', style='GO.TButton', command=lambda: goModelParamEst(gomodel, path, loaded))
    model2 = Button(paramTab, text="PNZ model", style='YamadaRay.TButton', command=lambda: pnzParamEst(pnz))
    model3 = Button(paramTab, text="Delayed S-shaped model", style='Delayed.TButton', command=lambda: delayedSParamEst(delayedS))
    model4 = Button(paramTab, text="Inflection S-shaped model", style='Inflection.TButton', command=lambda: inflectionSParamEst(inflectionS))
    model5 = Button(paramTab, text="Yamada Imperfect 1 model", style='YamadaIm1.TButton', command=lambda: yamadaImperfect1ParamEst(yamadaImperfect1))
    model6 = Button(paramTab, text="Yamada Imperfect 2 model", style='YamadaIm2.TButton', command=lambda: yamadaImperfect2ParamEst(yamadaImperfect2))
    model7 = Button(paramTab, text="Yamada Exponential model", style='YamadaExpo.TButton', command=lambda: yamadaExponentialParamEst(yamadaExponential))
    model8 = Button(paramTab, text="Vtub-Shaped model", style='Vtub.TButton', command=lambda: vtubParamEst(vtub))
    model9 = Button(paramTab, text="RMD model", style='RMD.TButton', command=lambda: rmdParamEst(rmd))
    model10 = Button(paramTab, text="Chang et al\'s model", style='Changs.TButton', command=lambda: changParamEst(changs))

    # placing model buttons
    (
        model1.place(relx= 0.1, rely=0.1, relwidth=0.35, relheight=0.08),
        model2.place(relx= 0.55, rely=0.1, relwidth=0.35, relheight=0.08),
        model3.place(relx= 0.1, rely=0.19, relwidth=0.35, relheight=0.08),
        model4.place(relx= 0.55, rely=0.19, relwidth=0.35, relheight=0.08),
        model5.place(relx= 0.1, rely=0.28, relwidth=0.35, relheight=0.08),
        model6.place(relx= 0.55, rely=0.28, relwidth=0.35, relheight=0.08),
        model7.place(relx= 0.1, rely=0.37, relwidth=0.35, relheight=0.08),
        model8.place(relx= 0.55, rely=0.37, relwidth=0.35, relheight=0.08),
        model9.place(relx= 0.1, rely=0.46, relwidth=0.35, relheight=0.08),
        model10.place(relx= 0.55, rely=0.46, relwidth=0.35, relheight=0.08)
    )

    # model ranking elements
    criteriaButton = Button(rankingTab, text="Show Calculated Criteria", command=rankModels)
    rankButton = Button(rankingTab, text="Calculate Ranks", command=rankModels)
    generateRankGraphButton = Button(rankingTab, text="Generate Rank Graph", command=rankModels(genGraph=True))
    estimatedParamsButton = Button(rankingTab, text="Estimated Model Parameters", command=paramEstimation)

    estimatedParamsButton.place(relx=0.2, rely=0.1, relwidth=0.6, relheight=0.2)
    criteriaButton.place(relx=0.2, rely=0.32, relwidth=0.6, relheight=0.2)
    rankButton.place(relx=0.2, rely=0.54, relwidth=0.6, relheight=0.2)
    generateRankGraphButton.place(relx=0.2, rely=0.76, relwidth=0.6, relheight=0.2)

    root.mainloop()


main()