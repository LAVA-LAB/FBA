function out = createTable(N,beta)

    tic
    prevtoc = toc;

    for k = 0:N/2
        disp("Compute for k = "+k)

        disp("Time difference: "+(toc-prevtoc))
        prevtoc = toc;

        [epsL(k+1), epsU(k+1)] = epsLU(k, N, beta);

        N_list(k+1) = N;
        k_list(k+1) = k;
        beta_list(k+1) = beta;
    end

    filename = "probabilityTable_N="+N+"_beta="+beta+".csv";
    writematrix([N_list' k_list' beta_list' epsL' epsU'], filename) 
    
end