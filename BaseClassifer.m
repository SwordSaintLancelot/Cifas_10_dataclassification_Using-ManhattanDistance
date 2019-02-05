classdef (Abstract) BaseClassifer < handle

    % Base class for all classifiers            

    

    methods(Abstract, Access = public)

        [maxscore, scores, timeElapsed] = predict(obj);

        

       
        [timeElapsed] = train(obj);

    end

    

end