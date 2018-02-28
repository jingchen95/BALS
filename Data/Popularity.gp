reset
set terminal postscript eps color solid enhanced 35
set terminal postscript eps monochrome 35
set size 2.0,1.1
set output "NTFX_ORI.eps"
unset key 

set style line 1  lt -1 lw 1 pt 0 ps 1
#set style line 2  lt -1 lw 4 pt 4 ps 1
#set style line 3  lt -1 lw 4 pt 8 ps 1
#set style line 4  lt 0 lw 4 pt 4 ps 1
#set style line 5  lt 0 lw 4 pt 8 ps 1
#set boxwidth 1 absolute
set style data histogram
#set style histogram cluster gap 3
#set style fill solid
set ylabel "Popularity (10^4)"
set xlabel "Movie ID"
set ytics ("0" 0, "5" 50000, "10" 100000, "15" 150000, "20" 200000, "25" 250000)
unset xtics
#set title "Netflix Movie Popularity"

plot 'NTFX_ORI.txt' using 2:xtic(1) fill pattern 1 lc rgb "#6699cc", 'NTFX.txt' using 2:xtic(1) fill pattern 1 lc rgb "#CC6666"
