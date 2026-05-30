// test-alignment.cpp: compile-guard for the LRC lyric-alignment code.
//
// The alignment code (dit-alignment-graph.h + lrc-alignment.h) is ported but NOT
// yet wired into the generation/cover post-processing path. This target only
// ensures it compiles + links against ACE's DiT primitives, so the eventual
// wiring starts from a known-good base. It does no real work.

#include "dit-alignment-graph.h"
#include "lrc-alignment.h"

int main() {
    // Reference an entry point from each header so they are not flagged unused.
    (void) (void *) &dit_alignment_extract;
    (void) (void *) &dtw_cpu;
    return 0;
}
