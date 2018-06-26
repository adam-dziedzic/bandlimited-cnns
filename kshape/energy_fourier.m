function F = energy_fourier(Data, etas)
% find first F of Data coeffs that exceeds etas/2

E = cumsum(abs(Data) .^ 2, 2);    % Energy is squared abs
E = bsxfun(@rdivide, E, E(:, end));
F = zeros(size(Data,1),length(etas));
for i = 1:size(F,1)
  m = floor(size(E, 2)/2) + 1;
  for j = 1:size(F, 2)
    F(i, j) = find(E(i, :) >= etas(j)/2, 1);   % find first F that exceeds eta/2
  end
end