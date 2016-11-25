%% Verdu 3.34 - single-user matched filter BER
numUsers = 5; % number of users
rhoVec = [-0.2 0.1 0.3 0.2]; % rho 1,2 to 1,5 - cross correlation values
AmpVec = [2 1 2 1]; % amplitudes relative to A1 (amplitude of user 1)

twoPowNList = 2.^[0:numUsers-2];
symCombList = ones(1,numUsers-1);

% generate Tx symbol combinations for all the users other than the user under test
for binListIdx = 1:2.^[numUsers-1] % for all symbol combinations
    for userIdx = 1:numUsers-1 % for all users
        if binListIdx > 1 % generate 2nd symbol combination onwards
            if (mod(binListIdx-1, twoPowNList(userIdx)) == 0) % condition to flip the sign of the symbol
                symCombList(binListIdx, userIdx) = -symCombList(binListIdx-1, userIdx); % flip the sign of the BPSK symbol
            else
                symCombList(binListIdx, userIdx) = symCombList(binListIdx-1, userIdx); % retain the sign of the BPSK symbol
            end
        end
    end
end

% count the number of closed eye conditions
closedEyeCnt = 0; % initialization
symAmpRhoProd = symCombList.*(ones(2^(numUsers-1), 1)*(AmpVec.*rhoVec)); % amplitude-rho product for all the symbol combinations
interferenceVec = sum(symAmpRhoProd.').'; % interference vector - for all the possible symbol combinations of other users

numBitErrConfirmed = length(find(interferenceVec > 1)); % cross boundary bit errors
numBitErrFlipCoin = length(find(interferenceVec == 1)); % on the boundary bit errors

bitErrRate = (numBitErrConfirmed + numBitErrFlipCoin/2)/(2^(numUsers-1)); % bit error rate

disp('Probability of Error of the single user matched filter for user 1 = ');
disp(bitErrRate);


