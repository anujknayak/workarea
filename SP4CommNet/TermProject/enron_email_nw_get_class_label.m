% Enron email network parsing
% Function to determine class labels for each nodes (persons related to the company)
function [classLabelList] = enron_email_nw_get_class_label()
% open and read file
fid = fopen('C:\Users\Anuj Nayak\Desktop\SigProc4CommNet\TermProject\EnronEmailNetwork\employees.txt','r');
inter = textscan(fid,'%[^\n]');
textFileStr = inter{1,1};
fclose(fid);

numEmployees = size(textFileStr, 1);

classLabelStrListUnique = {'Director','CEO','Vice President','President','Manager','Trader','Others'};
classLabelStrListUniqueActual = {'Director','CEO','President','Vice President','Manager','Trader','Others'}; % this vector is used only for sanity check
classLabelListLen = length(classLabelStrListUnique);
classLabelNumListUnique = [1 2 4 3 5 6 7];
classLabelList = zeros(184, 1);

% Extracting class-labels
for indEmp = 1:numEmployees % loop for all employees
    for indClassLabel = 1:classLabelListLen % for each employee, loop for all class labels
        strExistIdx = strfind(textFileStr{indEmp, 1}, classLabelStrListUnique{indClassLabel}); % check for class label string
        if ~isempty(strExistIdx) % if class label string is found, assign the class label number and break out of the inner loop
            classLabelList(indEmp) = classLabelNumListUnique(indClassLabel); % 
            textFileStr{indEmp, 1} = '';
            break;
        end
        if indClassLabel == classLabelListLen && classLabelList(indEmp) == 0;
            classLabelList(indEmp) = classLabelNumListUnique(7);
            textFileStr{indEmp, 1} = '';
        end        
    end
end

