% Programmer : Marvin A. Conn
% Army Researc Laboratory / Bowie State University
% 1/30/2020
% Do not dissiminate without author's permission.
% 
% Generate feature vectors  based on nandi papers, etc.
% input - complex waveform data(can be real also).
% ref1: 
% [1] O'Shea et al, Over-the-Air Deep Learning Based Radio Signal Classfication,
%    IEEE Journal of Selected Topics, 2018
% [2] A.K.Nandi, Automatic identification of digital modulation types
% [3] Aaron Smith, Michael Evans, NASA, "Modulation Classfiation of Sattelite 
%  Communication signals Using Cumulants and Neural Networks"
% NOTE: Accuracy of these computations require validation of accuracy.
function [FV,FVs] = computeNandiFeatVector(csignal)

% Special note!!!!
% normalizining input to unity input power / very important to do this
% step, otherwise cumulant/moment values do not seperate well. see one
% one of above references regarding this.
%
% [3] 
csignal = csignal/mean(abs(csignal).^2);

% Compute Moments
% compute p+q order moments - returns complex numbers if complex input.
[M20,M20p] = MPQfunc(2,0,csignal);
[M21,M21p] = MPQfunc(2,1,csignal);
[M40,M40p] = MPQfunc(4,0,csignal);
[M41,M41p]=  MPQfunc(4,1,csignal);
[M42,M42p] = MPQfunc(4,2,csignal);
[M43,M43p] = MPQfunc(4,3,csignal);
[M60,M60p] = MPQfunc(6,0,csignal);
[M61,M61p] = MPQfunc(6,1,csignal);
[M62,M62p] = MPQfunc(6,2,csignal);
[M63,M63p] = MPQfunc(6,3,csignal);
[M22] = MPQfunc(2,2,csignal); % needed for cumulant computation below

% Compute Cumulants - these formulas can be ontainedin literature.
C20 = M20; %? repeat of M20?
C21 = M21; %? repeat of M21?
C40 = M40 - 3*M20^2;
C41 = M40 - 3*M20*M21;
C42 = M42 - M20^2 - 2*M21^2;
C60 = M60 - 15*M20*M40 + 30*M20^3;
C61 = M61 - 5*M21*M40 - 10*M20*M41+30*M20^2*M21;
C62 = M62-6*M20*M42-8*M21*M41-M22*M40+6*M20^2*M22+24*M21^2*M20;
C63 = M63 - 9*M21*M42 + 12*M21^3 - 3*M20*M43-3*M22*M41+18*M20*M21*M22;

%Additional analog features II-A 
Meana = abs(mean(csignal));
Meanp = angle(mean(csignal)); %22
Std = std(csignal);
Kur = abs(KurtosisFunc(csignal)); % questionable calculation / check literature
anmax = AcnMax(csignal);
sigmaa = Sigma_aa(csignal);

% compute amplitudes of complex values
M20a=abs(M20);
M21a=abs(M21);
M40a=abs(M40);
M41a=abs(M41);
M42a=abs(M42);
M43a=abs(M43);
M60a=abs(M60);
M61a=abs(M61);
M62a=abs(M62);
M63a=abs(M63);
C20a=abs(C20);
C21a=abs(C21);
C40a=abs(C40);
C41a=abs(C41);
C42a=abs(C42);
C60a=abs(C60);
C61a=abs(C61);
C62a=abs(C62);
C63a=abs(C63);

% phases, not using for now
%     M20p,M21p,M40p,M41p,M42p,M43p,M60p,M61p,M62p,M63p,...

% returning two vectors
% FV = feature vector
% FSs= name of correcsponding feature in the vector
FV = [
    M20a,M21a,M40a,M41a,M42a,M43a,M60a,M61a,M62a,M63a,...
    C20a,C21a,C40a,C41a,C42a,C60a,C61a,C62a,C63a,...
    Meana,Meanp,Std,Kur,anmax,sigmaa];
FVs = [
    "M20a","M21a","M40a","M41a","M42a","M43a","M60a","M61a","M62a","M63a",...
    "C20a","C21a","C40a","C41a","C42a","C60a","C61a","C62a","C63a",...
    "Meana","Meanp","Std","Kur","anmax","sigmaa"];
end

% compute signal p, q moment.
function [Mpq,Mp] = MPQfunc(p, q, csignal)

    xpq = csignal.^(p-q);
    Mpq = mean(xpq.*((conj(csignal)).^q)); % returns complex value
    Mp = angle(Mpq);% phase angle
    
end

% pearson moment - wiki signal kurtosis
% need to compare this with matlab result!!!
% this normalizes the result with sigma
function KURT = KurtosisFunc(csignal)
  mu4 = mymoment(csignal,4);
  sig = std(csignal);
  KURT = mu4/(sig^4);  
end


% compute the centralized n order moment
% need to compare this with matlab result!!!
% I compared it, matlab returns the unnormalized
% moment, and so does this.
function mom = mymoment(csignal, n)
  % complex kurtotis
  m = mean(csignal);
  mom =  mean((csignal - m).^n);
end

% regarding red [2], I did not use some of their features
% because I don't have needed info those calcs require in
% the timefraem im to accomplish this.

% ref [2]
% instaneous amplitude - nandi 1995 paper
% normalized centered instaneous amplitude
% first feature
function anmax = AcnMax(csignal)

  %A = abs(hilbert(csignal)); % instantaneous amplitude
  A = abs(csignal);
  m = mean(A);  
  Acn = A / m - 1;
  anmax = max(abs(fft(Acn)).^2);
  
end

% ref [2]
% instaneous amplitude - nandi 1995 paper
% standard deviation of the absolute value
% of the normalized centered instaneous amplitude
% of the signal.
% 4th feature
function sigaa = Sigma_aa(csignal)
  
  N = numel(csignal); 
  A = abs(csignal);
  m = mean(A);
  Acn = A / m - 1;  
  s1 = sum(Acn.^2) / N;
  s2 = (sum(abs(Acn))/N)^2; 
  sigaa = sqrt(abs(s1-s2));
  
end


            



