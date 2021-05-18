beta = 0.1;

N_list = [25,50,100,200,400,800,1600,3200,6400];
N_list = [6400]

for N = 1:length(N_list)    
    createTable(N_list(N), beta)
end