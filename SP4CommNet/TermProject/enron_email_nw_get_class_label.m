% Enron email network parsing
% Function to determine class labels for each nodes (persons related to the company)
function [classLabelList] = enron_email_nw_get_class_label()
fid = fopen('C:\Users\Anuj Nayak\Desktop\SigProc4CommNet\TermProject\EnronEmailNetwork\employees.txt','r');
inter = textscan(fid,'%[^\n]');
textFileStr = inter{1,1};
fclose(fid);

numEmployees = size(textFileStr, 1);

classLabelStrListUnique = {'Director','CEO','Vice President','President','Manager','Trader','Others'};
classLabelListLen = length(classLabelStrListUnique);
classLabelNumListUnique = [1 2 4 3 5 6 7];
classLabelList = zeros(184, 1);

% Extracting class-labels
for indEmp = 1:numEmployees
    for indClassLabel = 1:classLabelListLen
        strExistIdx = strfind(textFileStr{indEmp, 1}, classLabelStrListUnique{indClassLabel});
        if ~isempty(strExistIdx)
            classLabelList(indEmp) = classLabelNumListUnique(indClassLabel);
            textFileStr{indEmp, 1} = '';
            break;
        end
        if indClassLabel == classLabelListLen && classLabelList(indEmp) == 0;
            classLabelList(indEmp) = classLabelNumListUnique(7);
            textFileStr{indEmp, 1} = '';
        end        
    end
end

