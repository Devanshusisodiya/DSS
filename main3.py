from tkinter import *
from tkinter import messagebox
from tkinter.font import Font
from tkinter.ttk import *
from tkinter.messagebox import *
from routines import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#------------------------------------------DATASET LOADER AND VIEWER--------------------------------------------
# SUBROUTINE TO LOAD DATA
def loadData(path, stringvar, loaded):
    try:
        data = pd.read_csv(path)
        stringvar.set(path)
        loaded.set(True)
        showinfo(title='Load successful', message='The dataset has been loaded successfuly.')
    except FileNotFoundError or PermissionError:
        return showerror(title="Incorrect path", message='The path entered does not exist. Please make sure to enter correct path.')

# WINDOW TO VIEW DATA
def viewData(path):
    #checking if the path entered or not
    if loaded.get():
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
    else:
        showerror(title="Dataset not loaded", message='Please enter the correct path of dataset and press the load button')

#-----------------------------------------------------------------------------------------------------------
#--------------MODEL SELECTOR, PARAMETER ESTIMATION AND RANK CALCULATION------------------------------------

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
        # rootDisp.geometry()
    
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
                paramString += parameter + "   :   " + parameterValue + "\n"
            # A LITTLE FORMATTING OF DATA
            model = model + "  "
            paramString = "\n" + paramString
            # ADDING MODEL AND PARAMETERS FOR DISPLAY IN THE WINDOW
            modelLabel = Label(rootDisp, text=model)
            modelParams = Label(rootDisp, text=paramString)
            modelLabel.grid(row=i, column=0, padx=25)
            modelParams.grid(row=i, column=1, padx=25)

        rootDisp.mainloop()
    pass

# THIS ROUTINE IS TO CALCULATE RANKS FOR MODELS
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
                print(model, "--->", param)
                if param:
                    data.append([model, param]) # THE FIRST ARG ---> STRING, SECOND ARG ---> PARAMETERS

            # GETTING THE DATA FROM DATASET IN ORDER TO CALCULATE PERFORMANCE CRITERIA
            dataset = pd.read_csv(path)
            X = dataset['Time']
            Y = dataset['CDF']

            # FOR CREATING ENTRIES IN data:
            rootRank = Toplevel(root)
            rootRank.title("Ranks")
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
                        pp = model.PP(params, X, Y)             
                        aic = model.aic(params, X, Y)
                        meop = model.meop(params, X, Y)
                        theil = model.theil(params, X, Y) 

                        criterias = [mse, mae, r2, adr, aic, pp, meop, theil] # WHILE ADDING COLUMNS TO THE TREEVIEW MAKE SURE TO ENTER THE COLUMNS IN THIS ORDER OF CRITERIA ONLY
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
            rankArr = [0 for _ in range(n)] # CHANGED HERE 
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

            rankArr = [[i] for i in rankArr] # CHANGED HERE
            # CALCULATED THE RANKS -----------------------------------------------

            modelsWithRank = []
            # criteriasWithRank = np.append(cdata, rankArr, 1)

            for i in range(len(rankArr)):
                row = [data[i][0], rankArr[i][0]]
                modelsWithRank.append(row)

            print(modelsWithRank)
            #-----------------------------------------------------------
            # ADDING TO TABLE
            columns = ('models', 'rank')
            columnsText = ('Models', 'Rank')
            tree = Treeview(rootRank, columns=columns, show='headings')

            for i in range(len(columns)):
                tree.heading(columns[i], text=columnsText[i])
                tree.column(columns[i])
            

            for i in modelsWithRank:
                # ADDING TO THE TREE VIEW
                tree.insert('', END, values=tuple(i))

            tree.grid(row=0, column=0)
            rootRank.mainloop()
        else:
            messagebox.showinfo(title="Technique Missing", message="Please select a technique to rank the models")

    r1 = Radiobutton(rootR, text='Entropy & TOPSIS', variable=var, value=1)
    submitButton = Button(rootR, text='Submit', command=lambda: calculateRanks(path))
    closeButton = Button(rootR, text="Close", command=rootR.destroy)

    r1.place(relx=0.2, rely=0.15)
    submitButton.place(relx=0.3, rely=0.3, relwidth=0.25, relheight=0.2)
    closeButton.place(relx=0.3, rely=0.5, relwidth=0.25, relheight=0.2)

    rootR.mainloop()

# THIS ROUTINE HERE IS TO CALCULATE ALL CRITERIAS FOR ALL MODELS
def computeC(path):

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
        rootCriteria = Toplevel(root)
        rootCriteria.title("Criteria Results")
        rootCriteria.geometry("995x300")
        rootCriteria.resizable(False, False)

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

        # ADDING TO TABLE
        columns = ('models', 'mse', 'mae', 'rsquare', 'adrsquare', 'aic', 'pp', 'meop', 'theil')
        columnsText = ('Models', 'MSE', 'MAE', 'R Squared', 'Adj. R Squared', 'AIC', 'PP', 'MEOP', 'TS')
        tree = Treeview(rootCriteria, columns=columns, show='headings')

        for i in range(len(columns)):
            tree.heading(columns[i], text=columnsText[i])
            tree.column(columns[i], minwidth=110, width=110)
        

        for i in criteriaData:
            # SHORTENING THE NUMBERS TILL 5 DECIMAL PLACES
            model = i[0]
            temp = [np.round_(j, decimals=5) for j in i[1:]]
            temp.insert(0, model)
            # ADDING TO THE TREE VIEW
            tree.insert('', END, values=tuple(temp))

        tree.place(relx=0, rely=0)

        # RANK BUTTON
        rankButton = Button(rootCriteria, text="Rank of Models", command=lambda: computeR(path))
        rankButton.place(relx=0.375, rely=0.775, relwidth=0.25, relheight=0.2)
        rootCriteria.mainloop()

# I DONT KNOW WHY I NAMED IT COMPUTE BUT ITS JUST THE MODEL PARAMETER ESTIMATOR
def compute(path):

    if loaded.get():
        data = pd.read_csv(path)

        X = data['Time']
        Y = data['CDF']

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

        rootComp = Toplevel(root)
        rootComp.geometry("800x500")
        rootComp.title("Models")

        canvas = Canvas(rootComp, width=800, height=500)
        # canvas.create_line()
        canvas.pack()

        # TESTING ---> IT WORKS BUT ONLY NEED TO APPLY ON SOME BUTTONS, NOT ALL
        style = Style()
        # style.theme_use('alt')
        style.configure('GO.TButton', background = 'red')
        style.configure('Delayed.TButton', background = 'red')
        style.configure('Inflection.TButton', background = 'red')
        style.configure('YamadaRay.TButton', background = 'red')
        style.configure('YamadaIm1.TButton', background = 'red')
        style.configure('YamadaIm2.TButton', background = 'red')
        style.configure('YamadaExpo.TButton', background = 'red')
        style.configure('Vtub.TButton', background = 'red')
        style.configure('RMD.TButton', background = 'red')
        style.configure('Changs.TButton', background = 'red')
        
        # I THINK I SHOULD ALSO INCLUDE THE PARAMETER ESTIMATION INSIDE THESE FUNCTIONS ONLY
        # WITHIN THE _sub METHOD, AND MAINTAIN THE DATABASE(GLOBAL MODEL DICTIONARY WITH THE ESTIMATED PARAMETERS)
        # IT IS GOING TO CONTAIN MULTIPLE FUNCTIONS IN ORDER TO INCLUDE MORE MODELS IN THE SYSTEM
        
        # BELOW IS FOR MODEL 1 ---> GO MODEL. ILL UPDATE THE NAMES LATER ON WHEN THE SYSTEM WILL BE WORKING FLAWLESSLY
        # NAMES HAVE BEEN CHANGED BECAUSE THE SYSTEM HAD NO TROUBLE INCORPORATING ANOTHER MODEL
        def goModelParamEst(modelObject):
            rootPar = Toplevel(rootComp)
            rootPar.title("GO Model Initial Parameters")
            rootPar.geometry("350x175")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get())
                ]
                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('GO.TButton', background='green')
                #
                rootPar.destroy()
                
                
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('GO.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

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

        # THIS IS FOR MODEL 2 ---> DELAYED S
        def delayedSParamEst(modelObject):
            rootPar = Toplevel(rootComp)
            rootPar.title("Delayed S Shaped Model Initial Parameters")
            rootPar.geometry("400x175")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get())
                ]
                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('Delayed.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('Delayed.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

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

        # THIS IS FOR MODEL 3 ---> INFLECTION S
        def inflectionSParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("Inflection S Shaped Initial Parameters")
            rootPar.geometry("400x200")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('Inflection.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('Inflection.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

            param1Label = Label(rootPar, text="a (cummulative faults)")  # THIS IS PARAMETER 'a' FOR INFLECTION S MODEL
            param2Label = Label(rootPar, text="b (detection rate)")  # THIS IS PARAMETER 'b' FOR INFLECTION S MODEL
            param3Label = Label(rootPar, text="\u03B2 (scale parameter)") #beta
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

        # THIS IS FOR MODEL 4 ---> YAMADA RAYLEIGH
        def yamadaRayleighParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("Yamada Rayleigh Model Initial Parameters")
            rootPar.geometry("400x250")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get()),
                    float(param4.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('YamadaRay.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('YamadaRay.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

            param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
            param2Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
            param3Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
            param4Label = Label(rootPar, text="\u03B3 (confidence level)") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
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

        # THIS IS FOR MODEL 5 ---> YAMADA IMPERFECT 1
        def yamadaImperfect1ParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("Yamada Imperfect1 Model Initial Parameters")
            rootPar.geometry("400x200")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('YamadaIm1.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('YamadaIm1.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

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

        # THIS IS FOR MODEL 6 ---> YAMADA IMPERFECT 2
        def yamadaImperfect2ParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("Yamada Imperfect2 Model Initial Parameters")
            rootPar.geometry("400x200")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('YamadaIm2.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('YamadaIm2.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

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

        # THIS IS FOR MODEL 7 ---> YAMADA EXPONENTIAL
        def yamadaExponentialParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("Yamada Exponential Model Initial Parameters")
            rootPar.geometry("400x250")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get()),
                    float(param4.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('YamadaExpo.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('YamadaExpo.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

            param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
            param2Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
            param3Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
            param4Label = Label(rootPar, text="\u03B3 (confidence level)") # THIS IS PARAMETER 'gamma' FOR YAMADA RAYLEIGH MODEL
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

        # THIS IS FOR MODEL 8 ---> VTUB SHAPED MODEL
        def vtubParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("VTub Model Initial Parameters")
            rootPar.geometry("400x275")

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
                # CHANGING BUTTON COLOR
                style.configure('Vtub.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('Vtub.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

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

        # THIS IS FOR MODEL 9 ---> RMD MODEL
        def rmdParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("RMD Model Initial Parameters")
            rootPar.geometry("400x250")

            def _sub():
                arrToOptimize = [
                    float(param1.get()),
                    float(param2.get()),
                    float(param3.get()),
                    float(param4.get())
                ]

                minimizationResults = minimize(fun=modelObject.OLS, x0=arrToOptimize, args=(X, Y), method='Nelder-Mead')
                modelMap[modelObject.name] = list(minimizationResults.x)
                # CHANGING BUTTON COLOR
                style.configure('RMD.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('RMD.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

            param1Label = Label(rootPar, text="a (cummulative faults)")      # THIS IS PARAMETER 'a' FOR YAMADA RAYLEIGH MODEL
            param2Label = Label(rootPar, text="b (detection rate)")      # THIS IS PARAMETER 'b' FOR YAMADA RAYLEIGH MODEL
            param3Label = Label(rootPar, text="\u03B1 (total expenditure)") # THIS IS PARAMETER 'alpha' FOR YAMADA RAYLEIGH MODEL
            param4Label = Label(rootPar, text="\u03B2 (scale parameter)") # THIS IS PARAMETER 'beta' FOR YAMADA RAYLEIGH MODEL
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
        
        # THIS IS FOR MODEL 10 ---> CHANGS MODEL
        def changParamEst(modelObject):    
            rootPar = Toplevel(rootComp)
            rootPar.title("Chang et al's Model Initial Parameters")
            rootPar.geometry("400x275")

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
                # CHANGING BUTTON COLOR
                style.configure('Changs.TButton', background='green')
                #
                rootPar.destroy()
            
            def _removeModel():
                try:
                    modelMap[modelObject.name] = None
                    # CHANGING BUTTON COLOR
                    style.configure('Changs.TButton', background='red')
                    #
                    rootPar.destroy()
                    messagebox.showinfo(title='Model Removal', message="Model has been successfuly removed")
                except:
                    messagebox.showinfo(title='Model Removal', message="Error occured while removing model")

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

        # ADDING BUTTONS TO INPUT MODEL PARAMETERS
        modelsLabel = Label(canvas, text="Models")
        model1 = Button(canvas, text='GO Model', style='GO.TButton', command=lambda: goModelParamEst(gomodel))
        model2 = Button(canvas, text="Yamada Rayleigh", style='YamadaRay.TButton', command=lambda: yamadaRayleighParamEst(yamadaR))
        model3 = Button(canvas, text="Delayed S", style='Delayed.TButton', command=lambda: delayedSParamEst(delayedS))
        model4 = Button(canvas, text="Inflection S", style='Inflection.TButton', command=lambda: inflectionSParamEst(inflectionS))
        model5 = Button(canvas, text="Yamada Imperfect 1", style='YamadaIm1.TButton', command=lambda: yamadaImperfect1ParamEst(yamadaImperfect1))
        model6 = Button(canvas, text="Yamada Imperfect 2", style='YamadaIm2.TButton', command=lambda: yamadaImperfect2ParamEst(yamadaImperfect2))
        model7 = Button(canvas, text="Yamada Exponential", style='YamadaExpo.TButton', command=lambda: yamadaExponentialParamEst(yamadaExponential))
        model8 = Button(canvas, text="Vtub Shaped", style='Vtub.TButton', command=lambda: vtubParamEst(vtub))
        model9 = Button(canvas, text="RMD", style='RMD.TButton', command=lambda: rmdParamEst(rmd))
        model10 = Button(canvas, text="Chang et al\'s", style='Changs.TButton', command=lambda: changParamEst(changs))

        # COMPUTATION BUTTONS
        computationLabel = Label(canvas, text="Calculation of Criterias & Rank")
        displayParameters = Button(canvas, text="Estimated Model Parameters", command=lambda: display())
        calculateCriteria = Button(canvas, text="Calculate Criterias", command=lambda: computeC(path))

        # ADDING PREVIOUSLY CREATED BUTTONS
        modelsLabel.place(relx=0.245, rely=0.01, relheight=0.05)
        model1.place(relx= 0.1, rely=0.08, relwidth=0.35, relheight=0.08)
        model2.place(relx= 0.1, rely=0.16, relwidth=0.35, relheight=0.08)
        model3.place(relx= 0.1, rely=0.24, relwidth=0.35, relheight=0.08)
        model4.place(relx= 0.1, rely=0.32, relwidth=0.35, relheight=0.08)
        model5.place(relx= 0.1, rely=0.4, relwidth=0.35, relheight=0.08)
        model6.place(relx= 0.1, rely=0.48, relwidth=0.35, relheight=0.08)
        model7.place(relx= 0.1, rely=0.56, relwidth=0.35, relheight=0.08)
        model8.place(relx= 0.1, rely=0.64, relwidth=0.35, relheight=0.08)
        model9.place(relx= 0.1, rely=0.72, relwidth=0.35, relheight=0.08)
        model10.place(relx= 0.1, rely=0.8, relwidth=0.35, relheight=0.08)

        # ADDING COMPUTATION BUTTONS
        computationLabel.place(relx= 0.625, rely=0.2)
        displayParameters.place(relx= 0.55, rely=0.3, relwidth=0.35, relheight=0.1)
        calculateCriteria.place(relx= 0.55, rely=0.4, relwidth=0.35, relheight=0.1)

        rootComp.mainloop()
    else:
        showerror(title="Dataset not loaded", message='Please enter the correct path of dataset and press the load button')

#------------------------------------------------------------------------------------------------------------------------

# MAIN FUNCTION
def main():
    global root
    global loaded

    root = Tk()
    root.title("Reliability DSS")
    root.geometry("500x250")

    path = StringVar()
    loaded = BooleanVar(False)
    # loaded.set(False)

    rootCanvas = Canvas(root, width=500, height=250)
    rootCanvas.pack()

    modelButton = Button(rootCanvas, text='Models & Computation', command=lambda: compute(pathEntry.get())) # SUBSTITUTE WITH THE models OVER HERE
    
    pathEntry = Entry(rootCanvas, width=35)
    loadButton = Button(rootCanvas, text='Load Dataset', command=lambda: loadData(pathEntry.get(), path, loaded))
    viewButton = Button(rootCanvas, text='View Dataset', command=lambda: viewData(pathEntry.get()))
    closeButton = Button(rootCanvas, text='Close', command=root.destroy)
    # PLACING THE BUTTONS  
    modelButton.place(relx=0.10, rely=0.35, relwidth=0.35, relheight=0.2)
    pathEntry.place(relx=0.50, rely=0.15)
    loadButton.place(relx=0.55, rely=0.25, relwidth=0.35, relheight=0.2)
    viewButton.place(relx=0.55, rely=0.45, relwidth=0.35, relheight=0.2)
    closeButton.place(relx=0.35, rely=0.75, relwidth=0.25, relheight=0.15)

    root.mainloop()

main()