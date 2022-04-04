import numpy as np
 
# ALL MODEL CLASSES
class GO:
    name = 'GO model'
    estAgain = False
    def model(self, a, b, t):
        '''Definition of GO Model'''
        return a*(1-np.exp(-b*t))
    
    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for GO Model'''
        y = self.model(C[0], C[1], X)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for GO Model'''
        y = self.model(params[0], params[1], X)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for GO Model'''
        y = self.model(params[0], params[1], X)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for GO Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT

    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for GO Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for GO Model'''
        y = self.model(params[0], params[1], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp

    def aic(self, params, X, Y):
        '''Calculates AIC for GO Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for GO Model'''
        y = self.model(params[0], params[1], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for GO Model'''
        residuals = np.square(self.model(params[0], params[1], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class DelayedS:
    name = 'Delayed-S Shaped model'
    estAgain = False
    def model(self, a, b, t):
        '''Definition of Delayed S Shaped Model'''
        return a*(1-(1+b*t)*np.exp(-b*t))
    
    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Delayed S Shaped Model'''
        x = X
        y = self.model(C[0], C[1], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Delayed S Shaped Model'''
        x = X
        y = self.model(params[0], params[1], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Delayed S Shaped Model'''
        x = X
        y = self.model(params[0], params[1], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos
    
    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Delayed S Shaped Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT

    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Delayed S Shaped Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Delayed S Shaped Model'''
        y = self.model(params[0], params[1], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp

    def aic(self, params, X, Y):
        '''Calculates AIC for Delayed S Shaped Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Delayed S Shaped Model'''
        y = self.model(params[0], params[1], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Delayed S Shaped Model'''
        residuals = np.square(self.model(params[0], params[1], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class InflectionS:
    name = 'Inflection-S Shaped model'
    estAgain = False
    def model(self, a, b, beta, t):
        '''Definition of Inflection S Shaped Model'''
        return a*(1-np.exp(-b*t))/(1+beta*np.exp(-b*t))

    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Inflection S Shaped Model'''
        x = X
        y = self.model(C[0], C[1], C[2], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Inflection S Shaped Model'''
        x = X
        y = self.model(params[0], params[1], params[2], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Inflection S Shaped Model'''
        x = X
        y = self.model(params[0], params[1], params[2], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Inflection S Shaped Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT
    
    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Inflection S Shaped Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Inflection S Shaped Model'''
        y = self.model(params[0], params[1], params[2], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp

    def aic(self, params, X, Y):
        '''Calculates AIC for Inflection S Shaped Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Inflection S Shaped Model'''
        y = self.model(params[0], params[1], params[2], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Inflection S Shaped Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class YamadaRayleigh:
    name = 'Yamada Rayleigh model'
    estAgain = False
    def model(self, a, alp, beta, gam, t):
        '''Definition of Yamada Rayleigh Model'''
        return a*(1-np.exp(-gam*alp*(1-np.exp(-beta*t**2/2))))

    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Yamada Rayleigh Model'''
        x = X
        y = self.model(C[0], C[1], C[2], C[3], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Yamada Rayleigh Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos

    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Yamada Rayleigh Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Yamada Rayleigh Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT
    
    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Yamada Rayleigh Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Yamada Rayleigh Model'''
        y = self.model(params[0], params[1], params[2], params[3], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp
    
    def aic(self, params, X, Y):
        '''Calculates AIC for Yamada Rayleigh Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Yamada Rayleigh Model'''
        y = self.model(params[0], params[1], params[2], params[3], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Yamada Rayleigh Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], params[3], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class YamadaImperfect1:
    name = 'Yamada Imperfect 1 model'
    estAgain = False
    def model(self, a, b, alp, t):
        '''Definition of Yamada Imperfect 1 Model'''
        return (a*b/(alp+b))*(np.exp(alp*t)-np.exp(-b*t))

    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Yamada Imperfect 1 Model'''
        x = X
        y = self.model(C[0], C[1], C[2], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Yamada Imperfect 1 Model'''
        x = X
        y = self.model(params[0], params[1], params[2], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Yamada Imperfect 1 Model'''
        x = X
        y = self.model(params[0], params[1], params[2], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Yamada Imperfect 1 Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT

    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Yamada Imperfect 1 Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Yamada Imperfect 1 Model'''
        y = self.model(params[0], params[1], params[2], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp

    def aic(self, params, X, Y):
        '''Calculates AIC for Yamada Imperfect 1 Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Yamada Imperfect 1 Model'''
        y = self.model(params[0], params[1], params[2], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Yamada Imperfect 1 Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class YamadaImperfect2:
    name = 'Yamada Imperfect 2 model'
    estAgain = False
    def model(self, a, b, alp, t):
        '''Definition of Yamada Imperfect 2 Model'''
        return a*(1-np.exp(-b*t))*(1-alp/b) + a*alp*t

    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Yamada Imperfect 2 Model'''
        x = X
        y = self.model(C[0], C[1], C[2], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Yamada Imperfect 2 Model'''
        x = X
        y = self.model(params[0], params[1], params[2], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos

    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Yamada Imperfect 2 Model'''
        x = X
        y = self.model(params[0], params[1], params[2], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Yamada Imperfect 2 Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT
    
    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Yamada Imperfect 2 Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Yamada Imperfect 2 Model'''
        y = self.model(params[0], params[1], params[2], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp
    
    def aic(self, params, X, Y):
        '''Calculates AIC for Yamada Imperfect 2 Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Yamada Imperfect 2 Model'''
        y = self.model(params[0], params[1], params[2], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Yamada Imperfect 2 Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class YamadaExponential:
    name = 'Yamada Exponential model'
    estAgain = False
    def model(self, a, alp, beta, gam, t):
        '''Definition of Yamada Exponential Model'''
        return a*(1-np.exp(-gam*alp*(1-np.exp(-beta*t))))

    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Yamada Exponential Model'''
        x = X
        y = self.model(C[0], C[1], C[2], C[3], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Yamada Exponential Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Yamada Exponential Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Yamada Exponential Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT
    
    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Yamada Exponential Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Yamada Exponential Model'''
        y = self.model(params[0], params[1], params[2], params[3], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp
    
    def aic(self, params, X, Y):
        '''Calculates AIC for Yamada Exponential Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Yamada Exponential Model'''
        y = self.model(params[0], params[1], params[2], params[3], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Yamada Exponential Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], params[3], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class Vtub:
    name = 'Vtub Shaped model'
    estAgain = False
    def model(self, a, b, alp, beta, n, t):
        '''Definition of Vtub Shaped Model'''
        return n*(1-(beta/(beta + a**(t**b) - 1))**alp)
    
    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Vtub Shaped Model'''
        x = X
        y = self.model(C[0],C[1],C[2],C[3],C[4], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Vtub Shaped Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], params[4], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Vtub Shaped Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], params[4], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Vtub Shaped Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], params[4], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT

    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Vtub Shaped Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], params[4], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Vtub Shaped Model'''
        y = self.model(params[0], params[1], params[2], params[3], params[4], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp
    
    def aic(self, params, X, Y):
        '''Calculates AIC for Vtub Shaped Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Vtub Shaped Model'''
        y = self.model(params[0], params[1], params[2], params[3], params[4], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Vtub Shaped Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], params[3], params[4], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class RMD:
    name = 'RMD model'
    estAgain = False
    def model(self, a, b, alp, beta, t):
        '''Definition of RMD Model'''
        return a*alp*(1-np.exp(-b*t)) - (a*b/(b-beta))*(np.exp(-beta*t) - np.exp(-b*t))
    
    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for RMD Model'''
        x = X
        y = self.model(C[0],C[1],C[2],C[3], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for RMD Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos

    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for RMD Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for RMD Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT
    
    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for RMD Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for RMD Model'''
        y = self.model(params[0], params[1], params[2], params[3], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp

    def aic(self, params, X, Y):
        '''Calculates AIC for RMD Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for RMD Model'''
        y = self.model(params[0], params[1], params[2], params[3], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for RMD Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], params[3], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

class Changs:
    name = 'Chang et al\'s model'
    estAgain = False
    def model(self, a, b, alp, beta, n, t):
        '''Definition of Chang et al's Model'''
        return n*(1-(beta/(beta + (a*t)**b))**alp)
    
    def OLS(self, C, X, Y):
        '''Ordinary Least Squares function to minimize and get parameters for Chang et al's Model'''
        x = X
        y = self.model(C[0],C[1],C[2],C[3],C[4], x)
        residuals = y-Y
        sos = np.sum(residuals**2)
        return sos

    def mse(self, params, X, Y):
        '''Calculates Mean Squared Error for Chang et al's Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], params[4], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        mos = np.sum(residuals**2) / (n-p)
        return mos
    
    def mae(self, params, X, Y):
        '''Calculates Mean Absolute Error for Chang et al's Model'''
        x = X
        y = self.model(params[0], params[1], params[2], params[3], params[4], x)
        residuals = y-Y
        n = len(Y)
        p = len(params)
        amos = np.sum(np.abs(residuals)) / (n-p)
        return amos

    def rsquare(self, params, X, Y):
        '''Calculates R Squared for Chang et al's Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], params[4], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        return 1-SSE/TOT
    
    def adrsquare(self, params, X, Y):
        '''Calculates Adjusted R Squared for Chang et al's Model'''
        SSE = np.sum((Y - self.model(params[0], params[1], params[2], params[3], params[4], X))**2)
        TOT = np.sum((Y - np.array([np.mean(Y) for _ in range(len(Y))]))**2)
        r2 = SSE/TOT
        n = len(Y)
        k = len(params)
        adr = 1 - r2*((n-1)/(n-k-1))
        return adr
    
    def PP(self, params, X, Y):
        '''Calculates Predictive Power for Chang et al's Model'''
        y = self.model(params[0], params[1], params[2], params[3], params[4], X)
        pp = np.sum(((y-Y)/Y)**2)
        return pp
    
    def aic(self, params, X, Y):
        '''Calculates AIC for Chang et al's Model'''
        n = len(Y)
        aic = n*np.log(self.mse(params, X, Y)) + 2*len(params)
        return aic

    def meop(self, params, X, Y):
        '''Calculates MEOP for Chang et al's Model'''
        y = self.model(params[0], params[1], params[2], params[3], params[4], X)
        residuals = y-Y
        adiff = np.sum(np.abs(residuals))
        denom = len(X) + 1
        return adiff / denom
    
    def theil(self, params, X, Y):
        '''Calculates Theil Index for Chang et al's Model'''
        residuals = np.square(self.model(params[0], params[1], params[2], params[3], params[4], X) - Y)
        numero = np.sqrt(np.sum(residuals))
        denom = np.sqrt(np.sum(np.square(Y)))
        return  (numero/denom)*100

# OBJECTS TO WORK WITH
gomodel = GO()
delayedS = DelayedS()
inflectionS = InflectionS()
yamadaR = YamadaRayleigh()
yamadaImperfect1 = YamadaImperfect1()
yamadaImperfect2 = YamadaImperfect2()
yamadaExponential = YamadaExponential()
vtub = Vtub()
rmd = RMD()
changs = Changs()