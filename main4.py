from pyexpat import model
import tkinter as tk
from tkinter import *
from tkinter.font import Font
from scipy.optimize import minimize
from tkinter.ttk import *
from tkinter.messagebox import *
from routines import *
import numpy as np
import pandas as pd

# routine to display estimated parameters
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
def computeR(path):
    rootR = Toplevel(root)
    rootR.title("Techniques")
    rootR.geometry("250x200")

    var = IntVar()
    def calculateRanks(path):
        if var.get() == 1:
            data = []
            # CHECKING IF MODEL PARAMETERS ARE EVEN ENTERED BY THE USER OR NOT, IF THEY ARE THE DATA LIST IS POPULATED
            # WITH MODEL ALONG WITH THE ESTIMATED PARAMETERS
            for model in modelMap:
                param = modelMap[model]
                if param:
                    data.append([model, param]) # THE FIRST ARG ---> STRING, SECOND ARG ---> PARAMETERS

            if len(data) < 2:
                showerror(title='Cannot run TOPSIS', message="None or less models have been selected, TOPSIS requires atleast 2 models.")
            else:
                # GETTING THE DATA FROM DATASET IN ORDER TO CALCULATE PERFORMANCE CRITERIA
                dataset = pd.read_csv(path)
                X = np.arange(1, len(dataset.Time)+1)
                Y = dataset['CDF']

                # FOR CREATING ENTRIES IN data:
                rootRank = Toplevel(root)
                rootRank.title("Ranks")
                rootRank.resizable(False, False)

                modelRoutines = [
                    gomodel,
                    pnz,
                    delayedS,
                    yamadaImperfect2,
                    vtub
                ]
                criteriaData = []


                # VALIDATING THE DATABASE KEYS WITH MODEL NAMES 
                for selected in data:
                    criterias = []
                    for model in modelRoutines:
                        if selected[0] == model.name:
                            params = selected[1]
                            mse = model.mse(params, X, Y)
                            r2 = model.rsquare(params, X, Y)           
                            aic = model.aic(params, X, Y)
                            bic = model.bic(params, X, Y)
                            pc = model.pc(params, X, Y)

                            criterias = [mse, r2, aic, bic, pc] # WHILE ADDING COLUMNS TO THE TREEVIEW MAKE SURE TO ENTER THE COLUMNS IN THIS ORDER OF CRITERIA ONLY
                            break
                    criterias.insert(0, selected[0])
                    criteriaData.append(criterias)

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

                print('W', w)

                # CALCULATED THE WEIGHT MATRIX -----------------------------------------------
                # CALCULATING THE RANK -------------------------------------------------------
                y = cdata / np.sqrt(np.sum(cdata**2, axis=0))
                v = w * y

                vpos = []
                vneg = []

                criteria_map = {
                }

                stringCriterias = ['mse', 'r2','aic', 'bic', 'pc']
                for i in range(len(stringCriterias)):
                    cr = stringCriterias[i]
                    criteria_map[cr] = v[:, i]

                maximizer = ['r2']
                minimizer = ['mse', 'aic', 'bic', 'pc']
            

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

                print('S+', spos)
                print('S-',sneg)

                # final relative closeness results
                c = sneg / (spos + sneg)

                print('C', c)

                # ranking the models
                ranked = {}
                rankArr = [0 for _ in range(n)] # CHANGED HERE 
                initialRank = 1

                for _ in range(n):
                    # populating rank list
                    index = np.argmax(c)
                    ranked[index] = initialRank
                    # discarding the considered index
                    c[index] = -1
                    initialRank += 1

                for i in range(n):
                    rankArr[i] = ranked[i]

                rankArr = [[i] for i in rankArr] # CHANGED HERE
                # CALCULATED THE RANKS -----------------------------------------------

                modelsWithRank = []

                for i in range(len(rankArr)):
                    row = [data[i][0], rankArr[i][0]]
                    modelsWithRank.append(row)

                print(modelsWithRank)
                print(w)
                #-----------------------------------------------------------
                # ADDING TO TABLE
                columns = ('models', 'rank')
                columnsText = ('Models', 'Rank')
                tree = Treeview(rootRank, columns=columns, show='headings')

                for i in range(len(columns)):
                    tree.heading(columns[i], text=columnsText[i])
                    tree.column(columns[i], anchor=CENTER)
                

                for i in modelsWithRank:
                    # ADDING TO THE TREE VIEW
                    tree.insert('', END, values=tuple(i))

                tree.grid(row=0, column=0)
                rootRank.mainloop()
        else:
            showinfo(title="Technique Missing", message="Please select a technique to rank the models")

    r1 = Radiobutton(rootR, text='Entropy & TOPSIS', variable=var, value=1)
    submitButton = Button(rootR, text='Submit', command=lambda: calculateRanks(path))
    closeButton = Button(rootR, text="Close", command=rootR.destroy)

    r1.place(relx=0.25, rely=0.15)
    submitButton.place(relx=0.35, rely=0.3, relwidth=0.25, relheight=0.2)
    closeButton.place(relx=0.35, rely=0.5, relwidth=0.25, relheight=0.2)

    rootR.mainloop()

# rountine to calculate criteria
def computeC(path):
    data = []
    # CHECKING IF MODEL PARAMETERS ARE EVEN ENTERED BY THE USER OR NOT, IF THEY ARE THE DATA LIST IS POPULATED
    # WITH MODEL ALONG WITH THE ESTIMATED PARAMETERS
    for model in modelMap:
        param = modelMap[model]
        if param:
            data.append([model, param]) # THE FIRST ARG ---> STRING, SECOND ARG ---> PARAMETERS

    if len(data) < 1:
        showerror(title='No models selected', message="None or less models have been selected, select atleast 1 model to view the criterias.")
    else:
        # GETTING THE DATA FROM DATASET IN ORDER TO CALCULATE PERFORMANCE CRITERIA
        dataset = pd.read_csv(path)
        X = np.arange(1, len(dataset.Time)+1)
        Y = dataset['CDF']

        # FOR CREATING ENTRIES IN data:
        rootCriteria = Toplevel(root)
        rootCriteria.title("Criteria Results")
        rootCriteria.geometry("700x300")
        rootCriteria.resizable(False, False)

        modelRoutines = [
            gomodel,
            pnz,
            delayedS,
            yamadaImperfect2,
            vtub
        ]
        criteriaData = []


        # VALIDATING THE DATABASE KEYS WITH MODEL NAMES 
        for selected in data:
            criterias = []
            for model in modelRoutines:
                if selected[0] == model.name:
                    params = selected[1]
                    mse = model.mse(params, X, Y)
                    r2 = model.rsquare(params, X, Y)           # IF ANY PROBLEM OCCURS IT IS IN PP, AIC, MEOP or TS  
                    aic = model.aic(params, X, Y)
                    bic = model.bic(params, X, Y)
                    pc = model.pc(params, X, Y)

                    criterias = [mse, r2, aic, bic, pc] # WHILE ADDING COLUMNS TO THE TREEVIEW MAKE SURE TO ENTER THE COLUMNS IN THIS ORDER OF CRITERIA ONLY
                    break
            criterias.insert(0, selected[0])
            criteriaData.append(criterias)

        # ADDING TO TABLE
        columns = ('models', 'mse', 'rsquare', 'aic', 'bic', 'pc')
        columnsText = ('Models', 'MSE', 'R\u00b2', 'AIC', 'BIC', 'PC')
        tree = Treeview(rootCriteria, columns=columns, show='headings')

        for i in range(len(columns)):
            if i == 0:
                tree.heading(columns[i], text=columnsText[i], anchor=CENTER)
                tree.column(columns[i], minwidth=145, width=145, anchor=CENTER)
            else:
                tree.heading(columns[i], text=columnsText[i], anchor=CENTER)
                tree.column(columns[i], minwidth=110, width=110, anchor=CENTER)
        

        for i in criteriaData:
            # SHORTENING THE NUMBERS TILL 5 DECIMAL PLACES
            model = i[0]
            temp = [np.round_(j, decimals=5) for j in i[1:]]
            temp.insert(0, model)
            # ADDING TO THE TREE VIEW
            tree.insert('', END, values=tuple(temp))

        tree.place(relx=0, rely=0)

        # RANK BUTTON
        rankButton = Button(rootCriteria, text="Click to calculate rank of models", command=lambda: computeR(path))
        rankButton.place(relx=0.375, rely=0.775, relwidth=0.25, relheight=0.2)
        rootCriteria.mainloop()


# main function to access the mainframe of the dss
def main():
    global root
    root = Tk()
    root.title("SRGM DSS")
    root.geometry("1240x500")
    myFont = Font(weight="bold")
    
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

    # subroutine to view the datase
    def viewDataset(stringvar):
        data = pd.read_csv(stringvar.get())
        # defining the treeview
        tree = Treeview(root, show='headings')
        # define columns
        tree['columns'] = ('time', 'cdf')
        tree.column('time', anchor=CENTER)
        tree.column('cdf', anchor=CENTER)
        # define headings
        tree.heading('time', text='Time', anchor=CENTER)
        tree.heading('cdf', text='CDF', anchor=CENTER)
        # add data to the treeview
        for i in range(len(data)):
            tree.insert('', END, values=(data.loc[i].Time, data.loc[i].CDF))
        
        # placing the tree on the canvas and removing the 
        # surrogate dataset canvas
        tree.place(relx=0.65, rely=0.45) 
        datasetCanvas.destroy()
    # subroutine to load the dataset
    def loadDataset(path, stringvar, loaded):
        try:
            data = pd.read_csv(path)
            stringvar.set(path)
            loaded.set(True)
            viewDataset(stringvar)
            showinfo(title='Load successful', message='The dataset has been loaded successfuly.')
        except FileNotFoundError or PermissionError:
            return showerror(title="Incorrect path", message='The path entered does not exist. Please make sure to enter correct path.')
    # below subroutines are written to estimate the parameters of the models
    # go model param estimation
    def goModelParamEst(modelObject, stringvar):
        rootPar = Toplevel(root)
        rootPar.title("GO Model Initial Parameters")
        rootPar.geometry("350x175")
        # getting the fitting data
        data = pd.read_csv(stringvar.get())
        X, Y = data.Time, data.CDF

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get())
            ]
            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # saving data
            mseMap[modelObject.name] = mse
            modelMap[modelObject.name] = list(minimizationResults.x)
           
            rootPar.destroy()
            
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")
        param2Label = Label(rootPar, text="b (detection rate)")
        param1 = Entry(rootPar, width=30) # THIS IS PARAMETER 'a' FOR GO MODEL
        param2 = Entry(rootPar, width=30) # THIS IS PARAMETER 'b' FOR GO MODEL
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.15)
        # ENTRY PLACING
        param1.place(relx=0.4, rely=0)
        param2.place(relx=0.4, rely=0.15)
        submitAndEstimate.place(relx=0.3, rely=0.5, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()
    # delayed s model param estimator
    def delayedSParamEst(modelObject, stringvar):
        rootPar = Toplevel(root)
        rootPar.title("Delayed S Shaped Model Initial Parameters")
        rootPar.geometry("400x175")
        # getting the fitting data
        data = pd.read_csv(stringvar.get())
        X, Y = data.Time, data.CDF

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get())
            ]
            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # saving data
            mseMap[modelObject.name] = mse
            modelMap[modelObject.name] = list(minimizationResults.x)
           
            rootPar.destroy()

            rootPar.destroy()
        
        def _removeModel():
            try:
                modelMap[modelObject.name] = None
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")
        param2Label = Label(rootPar, text="b (detection rate)")
        param1 = Entry(rootPar, width=35) # THIS IS PARAMETER 'a' FOR DELAYED S MODEL
        param2 = Entry(rootPar, width=35) # THIS IS PARAMETER 'b' FOR DELAYED S MODEL
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.15)
        # ENTRY PLACING
        param1.place(relx=0.4, rely=0)
        param2.place(relx=0.4, rely=0.15)
        submitAndEstimate.place(relx=0.3, rely=0.5, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)
        
        rootPar.mainloop()
    # pnz model param est
    def pnzParamEst(modelObject, stringvar):
        rootPar = Toplevel(root)
        rootPar.title("PNZ Model Initial Parameters")
        rootPar.geometry("400x250")
        # getting the fitting data
        data = pd.read_csv(stringvar.get())
        X, Y = data.Time, data.CDF

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
            # saving data
            mseMap[modelObject.name] = mse
            modelMap[modelObject.name] = list(minimizationResults.x)
           
            rootPar.destroy()
 
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b (detection rate)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar, width=35)
        param2 = Entry(rootPar, width=35)
        param3 = Entry(rootPar, width=35)
        param4 = Entry(rootPar, width=35)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.1)
        param3Label.place(relx=0, rely=0.2)
        param4Label.place(relx=0, rely=0.3)
        # ENTRY PLACING
        param1.place(relx=0.4, rely=0)
        param2.place(relx=0.4, rely=0.1)
        param3.place(relx=0.4, rely=0.2)
        param4.place(relx=0.4, rely=0.3)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()
    # pnz model param est
    def yamadaImperfect2ParamEst(modelObject, stringvar):
        rootPar = Toplevel(root)
        rootPar.title("Yamada Imperfect2 Model Initial Parameters")
        rootPar.geometry("400x200")
        # getting the fitting data
        data = pd.read_csv(stringvar.get())
        X, Y = data.Time, data.CDF

        def _sub():
            arrToOptimize = [
                float(param1.get()),
                float(param2.get()),
                float(param3.get())
            ]
            minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
            estimatesdParams = list(minimizationResults.x)
            mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
            # saving data
            mseMap[modelObject.name] = mse
            modelMap[modelObject.name] = list(minimizationResults.x)
           
            rootPar.destroy()
 
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA IMPERFECT 1 MODEL
        param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA IMPERFECT 1 MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA IMPERFECT 1 MODEL
        param1 = Entry(rootPar, width=35)
        param2 = Entry(rootPar, width=35)
        param3 = Entry(rootPar, width=35)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.125)
        param3Label.place(relx=0, rely=0.25)
        # ENTRY PLACING
        param1.place(relx=0.4, rely=0)
        param2.place(relx=0.4, rely=0.125)
        param3.place(relx=0.4, rely=0.25)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.70, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()
    # vtub model param est
    def vtubParamEst(modelObject, stringvar):
        rootPar = Toplevel(root)
        rootPar.title("VTub Model Initial Parameters")
        rootPar.geometry("400x275")
        # getting the fitting data
        data = pd.read_csv(stringvar.get())
        X, Y = data.Time, data.CDF

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
            # saving data
            mseMap[modelObject.name] = mse
            modelMap[modelObject.name] = list(minimizationResults.x)
           
            rootPar.destroy()
 
        def _removeModel():
            try:
                mseMap[modelObject.name] = np.inf
                modelMap[modelObject.name] = None
                rootPar.destroy()
                showinfo(title='Model Removal', message="Model has been successfuly removed")
            except:
                showinfo(title='Model Removal', message="Error occured while removing model")

        param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
        param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
        param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
        param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
        param5label = Label(rootPar, text="n")      # THIS IS PARAMETER 'n' FOR YAMADA RAYLEIGH MODEL
        param1 = Entry(rootPar, width=35)
        param2 = Entry(rootPar, width=35)
        param3 = Entry(rootPar, width=35)
        param4 = Entry(rootPar, width=35)
        param5 = Entry(rootPar, width=35)
        submitAndEstimate = Button(rootPar, text="Submit", command=_sub)
        removeModel = Button(rootPar, text="Remove", command=_removeModel)
        
        # LABELS PLACING
        param1Label.place(relx=0, rely=0)
        param2Label.place(relx=0, rely=0.09)
        param3Label.place(relx=0, rely=0.18)
        param4Label.place(relx=0, rely=0.27)
        param5label.place(relx=0, rely=0.36)
        # ENTRY PLACING
        param1.place(relx=0.4, rely=0)
        param2.place(relx=0.4, rely=0.09)
        param3.place(relx=0.4, rely=0.18)
        param4.place(relx=0.4, rely=0.27)
        param5.place(relx=0.4, rely=0.36)
        submitAndEstimate.place(relx=0.3, rely=0.50, relwidth=0.35, relheight=0.2)
        removeModel.place(relx=0.3, rely=0.7, relwidth=0.35, relheight=0.2)

        rootPar.mainloop()

    # computation buttons
    computeLabel = Label(root, text='Computation Processes')
    estParamsButton = Button(root, text='Estimated Parameters', command=display)
    criteriaButton = Button(root, text='Calculate Criteria', command=lambda: computeC(path.get()))
    rankButton = Button(root, text='Calculate Ranks', command=lambda: computeR(path.get()))
    # model buttons
    modelsLabel = Label(root, text='Model Selection')
    model1 = Button(root, text='GO model', command=lambda: goModelParamEst(gomodel, path))
    model2 = Button(root, text='PNZ model', command=lambda: pnzParamEst(pnz, path))
    model3 = Button(root, text='Yamada Imperfect 2 model', command=lambda: yamadaImperfect2ParamEst(yamadaImperfect2, path))
    model4 = Button(root, text='Delayed S-Shaped model', command=lambda: delayedSParamEst(delayedS, path))
    model5 = Button(root, text='V-tub Shaped model', command=lambda: vtubParamEst(vtub, path))

    # dataset elements
    datasetLabel = Label(root, text='Dataset Handling')
    pathEntryLabel = Label(root, text="Enter path of dataset")
    pathEntry = Entry(root, width=22)
    loadButton = Button(root, text='Load Dataset', command=lambda: loadDataset(pathEntry.get(), path, loaded))
    # dataset canvas
    datasetCanvas = Canvas(root, width=400, height=225, bg='#ffdac9')
    datasetMessage = tk.Label(datasetCanvas, text='Dataset Not Loaded.', bg='#ffdac9').place(relx=0.5, rely=0.5, anchor=CENTER)

    # placing compute labels
    computeLabel.place(relx=0.1, rely=0.1)
    modelsLabel.place(relx=0.45, rely=0.1)
    datasetLabel.place(relx=0.75, rely=0.1)
    # placing model buttons
    model1.place(relx= 0.37, rely=0.17, relwidth=0.25, relheight=0.1)
    model2.place(relx= 0.37, rely=0.27, relwidth=0.25, relheight=0.1)
    model3.place(relx= 0.37, rely=0.37, relwidth=0.25, relheight=0.1)
    model4.place(relx= 0.37, rely=0.47, relwidth=0.25, relheight=0.1)
    model5.place(relx= 0.37, rely=0.57, relwidth=0.25, relheight=0.1)
    # placing compute buttons
    estParamsButton.place(relx=0.03, rely=0.17, relwidth=0.25, relheight=0.1)
    criteriaButton.place(relx=0.03, rely=0.27, relwidth=0.25, relheight=0.1)
    rankButton.place(relx=0.03, rely=0.37 , relwidth=0.25, relheight=0.1)
    # placing dataset entry and butons dataset window
    (
        pathEntryLabel.place(relx=0.7, rely=0.2),
        pathEntry.place(relx=0.8, rely=0.2),
        loadButton.place(relx=0.7, rely=0.3, relwidth=0.25, relheight=0.1)
    )
    # placing the surrogate canvas for the dataset
    datasetCanvas.place(relx=0.65, rely=0.45)

    root.mainloop()

main()