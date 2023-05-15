    function y_encoded = label_encoding(y)
        y_encoded = zeros(size(y));
        for i = 1:numel(y)
            if strcmp(y{i}, 'setosa')
                y_encoded(i) = 1;
            else
                y_encoded(i) = -1;
            end
        end
    end