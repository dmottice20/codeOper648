% Value Iteration (VI) Algorithm, Puterman p.161
% Expected Total Discounted Reward Optimality Criterion

clc; % clear command window
clearvars; % clear variable memory
close all; % close any open graphs
% include folder with user-defined functions (e.g., limitdist)
% addpath('I:\MATLAB\MyFunctions'); THIS IS FOR THE GRAPH
tic; % start performance timer

%% Load External File
load('ARP_data')

%% Step 0 - Initialization, Preprocessing
% s, States -- age of car in one-month time periods
S = 1:120;
cardS = length(S);

% a \in A{s}, Action sets given state s
% at each state you have the option of keeping car for another month (a =
% 1) or trade in current car for i-month old car
cardA = 121; % cardinality of action set

for s = 1:cardS
    A{s} = 1:cardA; 
end

% Transition Prob Matrix for each action
P = cell(cardA,1);
for a = 1:cardA
    P{a} = zeros(cardS, cardS)
    if a == 1 % keep the car
        for s = 1:cardS
            for j = 1:cardS   
                if s < 119
                    if j == s+1
                        P{a}(s,j) = ARP_data(s+1, 4)
                    elseif j == 120
                        P{a}(s,j) = 1 - ARP_data(s+1, 4)
                    end
                elseif j == 120
                    P{a}(s,j) = 1
                end
            end
        end
    elseif a == 121 % buy 119-month old car
        for s = 1:cardS         
                P{a}(s,120) = 1
        end
    else 
        for s = 1:cardS % buy 0-118month old car       
                P{a}(s,a-1) = ARP_data(a-1, 4)
                P{a}(s,120) = 1-ARP_data(a-1, 4)
        end        
    end
end

% Reward matrix, reward obtained by selecting action a from state s
R = cell(cardA,1);
for a = 1:cardA % loop through each action
    R{a} = zeros(cardS, 1)
        if a == 1 % only incur operating cost for keeping car
            for s = 1:cardS
                R{a}(s) = -ARP_data(s+1, 3)
            end
        else
            for s = 1:cardS
                R{a}(s) = -ARP_data(a-1,1)+ARP_data(s+1,2)-ARP_data(a-1,3)
            end                
        end
end

%%% not sure if we need this
r = cell(cardA,1)
for a = 1:cardA
    r{a} = R{a}
end

toc; % end performance timer for loading problem data into memory

%% Step 1
tic; % start performance timer for VI algorithm execution
% Initialize value function, v_{n+1}(s)
v_np1 = zeros(cardS,1);

% Initialize decision rule vector, d(s)
d = zeros(cardS,1);

% Specify epsilon, error tolerance
epsilon = 0.01;

% Specify lambda, discount factor
lambda = 0.99;

% Specify convergence test criteria
convTest = epsilon*(1-lambda)/(2*lambda); % constant
convVal = inf; % set to \infty to pass initial while-loop check

% Specify iteration counter
n = 0;
% Specify maximum iterations
maxIter = 1500;

% For display in Figure 1, initialize table of values for
% successful toy, State 1, and convergence test values
inRunResultsTable = zeros(maxIter,2);

%% Steps 2-3 -- Repeatedly perform Bellman operation
% Execute while loop until error is small enough or number of iterations
% exceeds maximum number of iterations
while and(convVal >= convTest,n<maxIter)
% Update value function
v_n = v_np1;
% For each s \in S compute v_{n+1}^*(s)
for s = S
% Loop over action set, determine optimal action given state
% Initialize current best, v_{n+1}(s,a*)
v_np1_sa_Best = -inf;
for a = A{s}
% Compute v_{n+1}(s,a) for state-action pair
v_np1_sa = r{a}(s) + lambda*P{a}(s,:)*v_n;
if v_np1_sa > v_np1_sa_Best
% Update current best expected total reward
v_np1_sa_Best = v_np1_sa;
end
end
% Set v_{n+1}^(s) to best value attained
v_np1(s) = v_np1_sa_Best;
end
% Compute convergence criterion test value via infinity norm
convVal = norm(v_np1-v_n,inf);
% Record State 1 value and convergence test value
inRunResultsTable(n+1,:) = [v_np1(1) convVal];
% Increment iteration counter
n = n + 1;
end

%% Step 4 -- Obtain d*, optimal decision rule
% initialize transition probability matrix for d*
% useful for excursion analysis (e.g., computing steady state probability distributions)
Pd = zeros(cardS,cardS);
% Loop over state space

for s = S
% Loop over action space; determine optimal action given state s
% Initialize current best
v_np1_sa_Best = -inf;
for a = A{s}
% Compute v_{n+1}(s,a)
v_np1_sa = r{a}(s) + lambda*P{a}(s,:)*v_n;
if v_np1_sa > v_np1_sa_Best
% Update current best expected total reward and assoc. a*
v_np1_sa_Best = v_np1_sa;
d(s) = a;
end
end
% Set row s of Pd to row s of the P{a} matrix, where a = d(s)
Pd(s,:) = P{d(s)}(s,:);
end
toc % end performance timer for VI algorithm execution