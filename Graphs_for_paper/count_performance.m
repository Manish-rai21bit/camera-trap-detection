% Count Performance
figure('Renderer', 'painters', 'Position', [0 0 1600 400]);
x = categorical({'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10+'});
x = reordercats(x, {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10+'});
y = [37.2, 14.4, 9.4, 6.9, 4.9, 3.7, 2.6, 2.2, 1.5, 1.5, 15.6];
bar(x, y, 0.5);
xlabel('Animal count');
ylabel('Percentage');
ylim([0, 40]);
grid on;
title('Distribution of images with animal counts within');
text(1:length(y),y,strcat(num2str(y'),'%'),'vert','bottom','horiz','center');