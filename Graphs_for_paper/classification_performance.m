% This code generates graphs for section 3.1.2
% Overall Accuracy
% figure(1);
figure('Renderer', 'painters', 'Position', [5 5 850 600]);
x = [0, 1, 2];
y = [60.40, 70.00, 73.90];
bar(x, y, 0.5);
xlabel('Round number');
ylabel('Accuracy');
ylim([0, 100]);
grid on;
title('Overall Accuracy');
text(0:2,y,strcat(num2str(y'),'%'),'vert','bottom','horiz','center');

% Average of accuracy over all species
% figure(2);
figure('Renderer', 'painters', 'Position', [5 5 850 600]);
y = [67.00, 59.70, 58.70];
bar(x, y, 0.5);
xlabel('Round number');
ylabel('Accuracy');
ylim([0, 100]);
grid on;
title('Average of accuracy over all animals');
text(0:2,y,strcat(num2str(y'),'%'),'vert','bottom','horiz','center');

% Average of accuracy over all wilderbeest and zebra
% figure(3);
figure('Renderer', 'painters', 'Position', [5 5 850 600]);
y = [90.36, 95.11, 95.22];
bar(x, y, 0.5);
xlabel('Round number');
ylabel('Accuracy');
ylim([0, 100]);
grid on;
title('Average of accuracy over wildebeest and zebra');
text(0:2,y,strcat(num2str(y'),'%'),'vert','bottom','horiz','center');

% Average of accuracy over all wilderbeest and zebra
% figure(4);
figure('Renderer', 'painters', 'Position', [5 5 850 600]);
y = [89.29, 90.83, 92.41];
bar(x, y, 0.5);
xlabel('Round number');
ylabel('Accuracy');
ylim([0, 100]);
grid on;
title('Average of accuracy over top 5 species');
text(0:2,y,strcat(num2str(y'),'%'),'vert','bottom','horiz','center');