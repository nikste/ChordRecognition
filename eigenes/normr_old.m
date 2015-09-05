function n = normr_old(m)
%NORMR Normalize rows of matrix.
%
%	NORMR(M)
%	  M - a matrix.
%	Returns a matrix the same size with each
%	row normalized to a vector length of 1.
%
%	See also NORMC, PNORMC.

% Mark Beale, 1-31-92
% Copyright (c) 1992-97 by The MathWorks, Inc.
% $Revision: 1.3 $  $Date: 1997/05/14 22:10:50 $

if nargin < 1,error('Not enough input arguments.'); end

[mr,mc]=size(m);
if (mc == 1)
  n = m ./ abs(m);
else
  n=sqrt(ones./(sum((m.*m)')))'*ones(1,mc).*m;
end
