#include <MosaiCell/runtime.hh>

#include <ncurses.h>

#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("MosaiCell %s\n", mosaicell::version());
        return 0;
    }

    initscr();
    cbreak();
    noecho();
    printw("MosaiCell workbench runtime %s\n", mosaicell::version());
    printw("Native preprocessing backend: CellShard Blocked-ELL, fp16 values, fp32 QC accumulators.\n");
    printw("Press any key to exit.");
    refresh();
    getch();
    endwin();
    return 0;
}
