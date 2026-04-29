#include <CellShardPreprocess/runtime.hh>

#include <ncurses.h>

#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("CellShardPreprocess %s\n", cspre::version());
        return 0;
    }

    initscr();
    cbreak();
    noecho();
    printw("CellShardPreprocess workbench runtime %s\n", cspre::version());
    printw("Native preprocessing backend: CellShard Blocked-ELL, fp16 values, fp32 QC accumulators.\n");
    printw("Press any key to exit.");
    refresh();
    getch();
    endwin();
    return 0;
}
