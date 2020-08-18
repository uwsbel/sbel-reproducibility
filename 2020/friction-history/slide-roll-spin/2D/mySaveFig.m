function mySaveFig(num, figname)

if strcmp(computer, 'GLNXA64') == true % linux system
    fig_directory = '/home/luning/Source/projectlets/friction-contact/slide-roll-spin/results/figsForPapers/';
    png_directory = '/home/luning/Papers/2020/FrictionHistory/Images/';
end

if strcmp(computer, 'MACI64') == true % mac
    fig_directory = '/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/results/figsForPapers/';
    png_directory = '/Users/luning/Papers/2020/FrictionHistory_Journal/Images/';
end

figure(num)
print(gcf, strcat(png_directory, figname, '.png'), '-dpng', '-r300');
savefig(strcat(fig_directory, figname));
