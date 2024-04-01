def saturation_9(In,S,St,varargin)
if size(varargin,2) == 0
    out = In.*smoothThreshold_storage_logistic(S,St);
elseif size(varargin,2) == 1
    out = In.*smoothThreshold_storage_logistic(S,St,varargin(1));
elseif size(varargin,2) == 2
    out = In.*smoothThreshold_storage_logistic(S,St,varargin(1),varargin(2));    

