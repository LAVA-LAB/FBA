function [epsL, epsU] = epsLU(k,N,bet)
    % Compute inverse beta function to guide initial choie of t1 and t2
    alphaL = betaincinv(bet,k,N-k+1); 
    alphaU = 1-betaincinv(bet,N-k+1,k); 
    
    % Compute combination (i, k) in logarithmic base (term 1)
    m1 = [k:1:N]; 
    aux1 = sum(triu(log(ones(N-k+1,1)*m1),1),2); 
    aux2 = sum(triu(log(ones(N-k+1,1)*(m1-k)),1),2); 
    coeffs1 = aux2-aux1; 
    
    % Compute combination (i, k) in logarithmic base (term 2)
    m2 = [N+1:1:4*N]; 
    aux3 = sum(tril(log(ones(3*N,1)*m2)),2); 
    aux4 = sum(tril(log(ones(3*N,1)*(m2-k))),2); 
    coeffs2 = aux3-aux4;

    % Initial guess for value of t (for lower bound epsilon)
    t1 = 1-alphaL; 
    t2 = 1;
    
    % Initialize bisection problem
    poly1 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1')*log(t1)))... 
        -bet/(6*N)*sum(exp(coeffs2 + (m2'-N)*log(t1)));
    poly2 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1')*log(t2)))... 
        -bet/(6*N)*sum(exp(coeffs2 + (m2'-N)*log(t2)));

    % Loop over bisection problem until desired precision is reached
    if ((poly1*poly2) > 0) 
        epsL = 0;
    else
        while t2-t1 > 1e-10 
            t = (t1+t2)/2; 
            polyt = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1')*log(t)))... 
                -bet/(6*N)*sum(exp(coeffs2 + (m2'-N)*log(t)));
            if polyt > 0
                t1=t;
            else
                t2=t;
            end
        end
        epsL = 1-t2;
    end
    
    % Initial guess for value of t (for upper bound epsilon)
    t1 = 0; 
    t2 = 1-alphaU; 
    
    % Initialize bisection problem
    poly1 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1')*log(t1)))...
        -bet/(6*N)*sum(exp(coeffs2 + (m2'-N)*log(t1)));
    poly2 = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1 - (N-m1')*log(t2)))... 
        -bet/(6*N)*sum(exp(coeffs2 + (m2'-N)*log(t2)));
    
    % Loop over bisection problem until desired precision is reached
    if ((poly1*poly2) > 0) 
        epsU = 0;
    else
        while t2-t1 > 1e-10 
            t = (t1+t2)/2; 
            polyt = 1+bet/(2*N)-bet/(2*N)*sum(exp(coeffs1-(N-m1')*log(t)))... 
                -bet/(6*N)*sum(exp(coeffs2 + (m2'-N)*log(t)));
            if polyt > 0 
                t2=t;
            else
                t1=t;
            end
        end
        epsU = 1-t1;
    end 
end