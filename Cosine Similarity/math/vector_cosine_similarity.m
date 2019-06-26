function [similarity] = vector_cosine_similarity(a,b)
%   VECTOR_COSINE_SIMILARITY  measures similarity between two non-zero vectors using an inner product space that measures the cosine of the angle between them.
%   a is non-zero vector of length N
%   b is non-zero vector of length N
%
%   similarity is a scalar which represent the similarity of a and b
%   ranging from [-1,1]

    similarity = dot(a,b) / (norm(a) * norm(b));
end
