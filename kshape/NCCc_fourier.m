function cc_sequence = NCCc_fourier(x,y)
% cc_sequence = NCCc_fourier(x,y) 
% Input:
%   x, y are fourier cofficients, presumably of zero-padded time series
% Output: 
%   cc_sequence is normalized cross-correlation column vector

len = length(x)/2;   % original ts right zero-padded, so only need left half
if isrow(x)
    x=x.';   % nonconjugate transpose
end
if isrow(y)
    y=y.';   % nonconjugate transpose
end

x = x ./ norm(x);
y = y ./ norm(y);

r = real(ifft(x .* conj(y)));
r = [r(end-len+2:end) ; r(1:len)];
%cc_sequence = length(x)*r./((norm(x)+norm(y)));
cc_sequence = length(x)*r;
