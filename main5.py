from threading import Timer
from tkinter.messagebox import *
from tkinter import *
from tkinter.ttk import *
import numpy as np
import pandas as pd
import tkinter as tk
from scipy.optimize import dual_annealing, minimize
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
def rankModels(path, loaded, genGraph = False):
    # routine to get the weights
    def getWeights(weightOp, cdata):
        if weightOp.get() == 1:
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
            
            return w
        elif weightOp.get() == 2:
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
            
            return w
        elif weightOp.get() == 3:
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
            
            return w
    # routine to get the ranks
    def getRanks(rankOp, weightVec, cdata):
        if rankOp.get() == 1:
            n = cdata.shape[0]
            y = cdata / np.sqrt(np.sum(cdata**2, axis=0))
            v = weightVec * y

            vpos = []
            vneg = []

            criteria_map = {
            }

            stringCriterias = ['mse', 'mae', 'r2', 'adjr2','aic', 'bic', 'pc', 'pp', 'meop', 'theil',]
            for i in range(len(stringCriterias)):
                cr = stringCriterias[i]
                criteria_map[cr] = v[:, i]

            for i in criteria_map:
                print(i, criteria_map[i])

            maximizer = ['r2', 'adjr2']
            minimizer = ['mse', 'mae', 'pp', 'meop', 'aic', 'bic', 'pc', 'theil']
        

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
            
            return modelsWithRank
        elif rankOp.get() == 2:
            return
    
    if loaded.get():
        data = pd.read_csv(path.get())
        X = data.Time
        Y = data.CDF
        # compute
        data = []
            # CHECKING IF MODEL PARAMETERS ARE EVEN ENTERED BY THE USER OR NOT, IF THEY ARE THE DATA LIST IS POPULATED
            # WITH MODEL ALONG WITH THE ESTIMATED PARAMETERS
        for model in modelMap:
            param = modelMap[model]
            if param:
                data.append([model, param])
        
        if len(data) >= 2:
                if not genGraph:
                    rootOps = Toplevel(root)
                    rootOps.geometry("300x220")

                    weightOp = IntVar()
                    rankOp = IntVar()

                    modelRoutines = [
                        gomodel,
                        pnz,
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
                                bic = model.bic(params, X, Y)
                                pc = model.pc(params, X, Y)
                                meop = model.meop(params, X, Y)
                                theil = model.theil(params, X, Y) 

                                criterias = [mse, mae, r2, adr, aic, bic, pc, pp, meop, theil] # WHILE ADDING COLUMNS TO THE TREEVIEW MAKE SURE TO ENTER THE COLUMNS IN THIS ORDER OF CRITERIA ONLY
                                break
                        criterias.insert(0, selected[0])
                        criteriaData.append(criterias)
                    # CALCULATE TOPSIS OVER HERE--------------------------------
                    cdata = []
                    for modelData in criteriaData:
                        cdata.append(modelData[1:])
                    cdata = np.array(cdata)

                    # routine to return rank window
                    def rankCalc(weightOp, rankOp, cdata):
                        weights = getWeights(weightOp, cdata)
                        ranks = getRanks(rankOp, weights, cdata)

                        rootRank = Toplevel(rootOps)
                        rootRank.resizable(False, False)

                        columns = ('models', 'rank')
                        columnsText = ('Models', 'Rank')
                        tree = Treeview(rootRank, columns=columns, show='headings')

                        for i in range(len(columns)):
                            tree.heading(columns[i], text=columnsText[i])
                            tree.column(columns[i], anchor=CENTER)
                        

                        for i in ranks:
                            # ADDING TO THE TREE VIEW
                            tree.insert('', END, values=tuple(i))

                        tree.grid(row=0, column=0)

                        rootRank.mainloop()

                    w1 = Radiobutton(rootOps, text='AHP', value=1, variable=weightOp)
                    w2 = Radiobutton(rootOps, text='EWM', value=2, variable=weightOp)
                    r1 = Radiobutton(rootOps, text='CODAS', value=1, variable=rankOp)
                    r2 = Radiobutton(rootOps, text='TOPSIS', value=2, variable=rankOp)
                    submit = Button(rootOps, text='Submit', command=lambda: rankCalc(weightOp, rankOp, cdata))

                    w1.grid(row=0, column=0)
                    w2.grid(row=1, column=0)
                    r1.grid(row=0, column=2)
                    r2.grid(row=1, column=2)
                    submit.grid(row=2, column=1)

                    rootOps.mainloop()
    
                else:
                    pass
        else:
            showerror(title='Cannot rank', message="None or less models have been selected, ranking requires atleast 2 models.")
 
# routine to calculate criteria
def criteria(path):
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
        rootCriteria.geometry("1250x230")
        rootCriteria.resizable(False, False)

        modelRoutines = [
            gomodel,
            pnz,
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
                    bic = model.bic(params, X, Y)
                    pc = model.pc(params, X, Y)
                    meop = model.meop(params, X, Y)
                    theil = model.theil(params, X, Y)

                    criterias = [mse, mae, r2, adr, aic, bic, pc, pp, meop, theil] # WHILE ADDING COLUMNS TO THE TREEVIEW MAKE SURE TO ENTER THE COLUMNS IN THIS ORDER OF CRITERIA ONLY
                    break
            criterias.insert(0, selected[0])
            criteriaData.append(criterias)

        # ADDING TO TABLE
        columns = ('models', 'mse', 'mae', 'rsquare', 'adrsquare', 'aic', 'bic', 'pc', 'pp', 'meop', 'theil')
        columnsText = ('Models', 'MSE', 'MAE', 'R\u00b2', 'Adj. R\u00b2', 'AIC', 'BIC', 'PC', 'PP', 'MEOP', 'TS')
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

        rootCriteria.mainloop()

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
            X = data.Time
            Y = data.CDF
            # rest compute
            rootPar = Toplevel(paramTab)
            rootPar.title("GO Model Initial Parameters")
            rootPar.geometry("550x275")

            def _sub():
                
                optimMessage['text'] = 'optimizing...\ndo not close this window'

                def optim():

                    bounds = [
                        (float(param1lb.get()), float(param1ub.get())),
                        (float(param2lb.get()), float(param2ub.get()))
                    ]

                    minimizationResults = dual_annealing(func=modelObject.OLS, args=(X, Y), bounds=bounds, maxiter=int(iterEntry.get()))
                    
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
                t = Timer(1, optim)
                t.start()
                
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
            #
            optimFrame = Canvas(rootPar, width=250, height=50, bg='#ffdac9')
            optimMessage = tk.Label(optimFrame, text='Enter the bounds and click submit', bg='#ffdac9')
            #
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
            # OPTIM FRAME PLACING
            optimFrame.place(relx=0.3, rely=0)
            optimMessage.place(relx=0.15, rely=0.2)
            # LABELS PLACING
            param1Label.place(relx=0, rely=0.25)
            param2Label.place(relx=0, rely=0.35)
            # ENTRY PLACING
            param1lb.place(relx=0.4, rely=0.25)
            param2lb.place(relx=0.4, rely=0.35)
            param1ub.place(relx=0.6, rely=0.25)
            param2ub.place(relx=0.6, rely=0.35)
            submitAndEstimate.place(relx=0.3, rely=0.5, relwidth=0.35, relheight=0.15)
            removeModel.place(relx=0.3, rely=0.65, relwidth=0.35, relheight=0.15)

            rootPar.mainloop()
        else:
            showinfo(title='Dataset not loaded yet', message='Please load the dataset before model selection')
    
    # THIS IS FOR MODEL 2 ---> DELAYED S
    def delayedSParamEst(modelObject, path, loaded):
        if loaded.get():
            # getting the data
            data = pd.read_csv(path.get())
            X = data.Time
            Y = data.CDF
            # rest compute
            rootPar = Toplevel(paramTab)
            rootPar.title("Delayed S Shaped Model Initial Parameters")
            rootPar.geometry("550x275")

            def _sub():

                optimMessage['text'] = 'optimizing...\ndo not close this window'
                
                def optim():
                    
                    bounds = [
                        (float(param1lb.get()), float(param1ub.get())),
                        (float(param2lb.get()), float(param2ub.get()))
                    ]

                    minimizationResults = dual_annealing(func=modelObject.OLS, args=(X, Y), bounds=bounds, maxiter=int(iterEntry.get()))

                    estimatesdParams = list(minimizationResults.x)
                    mse = np.round_(modelObject.mse(estimatesdParams, X, Y), decimals=4)
                    # CHECKING WETHER CALCULATED MSE IS SMALLER THAN THE STORED MSE/ PREVIOUS MSE
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
                    style.configure('Delayed.TButton', background='green')
                    #
                    rootPar.destroy()            
                t = Timer(1, optim)
                t.start()

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
            

            optimFrame = Canvas(rootPar, width=250, height=50, bg='#ffdac9')
            optimMessage = tk.Label(optimFrame, text='Enter the bounds and click submit', bg='#ffdac9')

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
            # OPTIM FRAME PLACING
            optimFrame.place(relx=0.3, rely=0)
            optimMessage.place(relx=0.15, rely=0.2)
            # LABELS PLACING
            param1Label.place(relx=0, rely=0.25)
            param2Label.place(relx=0, rely=0.35)
            # ENTRY PLACING
            param1lb.place(relx=0.4, rely=0.25)
            param2lb.place(relx=0.4, rely=0.35)
            param1ub.place(relx=0.6, rely=0.25)
            param2ub.place(relx=0.6, rely=0.35)
            submitAndEstimate.place(relx=0.3, rely=0.5, relwidth=0.35, relheight=0.15)
            removeModel.place(relx=0.3, rely=0.65, relwidth=0.35, relheight=0.15)

            rootPar.mainloop()
        else:
            showinfo(title='Dataset not loaded', message='load dataset')

    # THIS IS FOR MODEL 3 ---> INFLECTION S
    def inflectionSParamEst(modelObject, path, loaded):
        if loaded.get():

            data = pd.read_csv(path.get())
            X = data.Time
            Y = data.CDF

            rootPar = Toplevel(paramTab)
            rootPar.title("Inflection S Shaped Initial Parameters")
            rootPar.geometry("550x200")

            def _sub():

                def optim():
                    bounds = [
                        (float(param1lb.get()), float(param1ub.get())),
                        (float(param2lb.get()), float(param2ub.get())),
                        (float(param3lb.get()), float(param3ub.get()))
                    ]

                    minimizationResults = dual_annealing(func=modelObject.OLS, args=(X, Y), bounds=bounds, maxiter=int(iterEntry.get()))

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
                t = Timer(1, optim)
                t.start()
                
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
        else:
            showinfo()

    # THIS IS FOR MODEL 4 ---> PHAM NORDMANN ZHANG
    def pnzParamEst(modelObject, path, loaded):
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
    def yamadaImperfect1ParamEst(modelObject, path, loaded):
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
    def yamadaImperfect2ParamEst(modelObject, path, loaded):
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
    def yamadaExponentialParamEst(modelObject, path, loaded):
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
    def vtubParamEst(modelObject, path, loaded):
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
    def rmdParamEst(modelObject, path, loaded):
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
    def changParamEst(modelObject, path, loaded):
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
    model3 = Button(paramTab, text="Delayed S-shaped model", style='Delayed.TButton', command=lambda: delayedSParamEst(delayedS, path, loaded))
    model4 = Button(paramTab, text="Inflection S-shaped model", style='Inflection.TButton', command=lambda: inflectionSParamEst(inflectionS))
    model5 = Button(paramTab, text="Yamada Imperfect 1 model", style='YamadaIm1.TButton', command=lambda: yamadaImperfect1ParamEst(yamadaImperfect1))
    model6 = Button(paramTab, text="Yamada Imperfect 2 model", style='YamadaIm2.TButton', command=lambda: yamadaImperfect2ParamEst(yamadaImperfect2))
    model7 = Button(paramTab, text="Yamada Exponential model", style='YamadaExpo.TButton', command=lambda: yamadaExponentialParamEst(yamadaExponential))
    model8 = Button(paramTab, text="Vtub-Shaped model", style='Vtub.TButton', command=lambda: vtubParamEst(vtub))
    model9 = Button(paramTab, text="RMD model", style='RMD.TButton', command=lambda: rmdParamEst(rmd))
    model10 = Button(paramTab, text="Chang et al\'s model", style='Changs.TButton', command=lambda: changParamEst(changs))

    iterLabel = Label(paramTab, text='Enter number of iterations:')
    iterEntry = Entry(paramTab, width=30)

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
        model10.place(relx= 0.55, rely=0.46, relwidth=0.35, relheight=0.08),

        iterLabel.place(relx= 0.25, rely=0.6),
        iterEntry.place(relx= 0.5, rely=0.6),
        # iterButton.place(relx= 0.3, rely=0.7, relwidth=0.35, relheight=0.08)
    )

    # model ranking elements
    criteriaButton = Button(rankingTab, text="Show Calculated Criteria", command=lambda: criteria(path.get()))
    rankButton = Button(rankingTab, text="Calculate Ranks", command=lambda: rankModels(path=path, loaded=loaded, genGraph=False))
    generateRankGraphButton = Button(rankingTab, text="Generate Rank Graph", command=rankModels(path=path, loaded=loaded, genGraph=True))
    estimatedParamsButton = Button(rankingTab, text="Estimated Model Parameters", command=paramEstimation)

    estimatedParamsButton.place(relx=0.2, rely=0.1, relwidth=0.6, relheight=0.2)
    criteriaButton.place(relx=0.2, rely=0.32, relwidth=0.6, relheight=0.2)
    rankButton.place(relx=0.2, rely=0.54, relwidth=0.6, relheight=0.2)
    generateRankGraphButton.place(relx=0.2, rely=0.76, relwidth=0.6, relheight=0.2)

    root.mainloop()


main()