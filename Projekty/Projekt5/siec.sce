clear close()
P = -2 :0.1: 2;
T = P.^2 + 1*(rand(P)-0.5); 

//siec 
S1 = 3;
W1 = rand(S1, 1)- 0.5;
B1 = rand(S1, 1)- 0.5;
W2 = rand(1, S1) -0,5;
B2 = rand(1,1) -0.5;
lr = 0.01


for  epoka = 1 : 1000
    //odpowiedz sieci
    s = W1*P + B1*ones(P)
    A1 = atan(s);
    A2 = W2*A1 + B2;
    
    //propagacja wsteczna
    E2 = T - A2;
    E1 = W2'*E2;
    
    dW2 = lr* E2 * A1';
    dB2 = lr *E2 * ones(E2)';
    dW1 = lr * (1./(ones(s) + s.^2)) .* E1 * P';
    dB1 = lr * (1./(ones(s) + s.^2)) .* E1 * ones(P)';
    
    W2 = W2 + dW2;
    B2 =  B2 + dB2;
    W1 = W1 + dW1;
    B1 = B1 + dB1;
    
    
    if modulo(epoka, 5)==0 then
        clf();
        plot(P,T, 'r*')
        plot(P,A2)
        sleep(50);
    end

end
