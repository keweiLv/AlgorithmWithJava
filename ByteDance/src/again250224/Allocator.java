package again250224;

import java.util.Arrays;

/**
 * 设计内存分配器
 */
public class Allocator {

    private final int[] memory;

    public Allocator(int n) {
        memory = new int[n];
    }

    public int allocate(int size, int mID) {
        int free = 0;
        for (int i = 0; i < memory.length; i++) {
            if (memory[i] > 0) {
                free = 0;
                continue;
            }
            free++;
            if (free == size) {
                Arrays.fill(memory, i - size + 1, i + 1, mID);
                return i - size + 1;
            }
        }
        return -1;
    }

    public int freeMemory(int mID) {
        int ans = 0;
        for (int i = 0; i < memory.length; i++) {
            if (memory[i] == mID) {
                ans++;
                memory[i] = 0;
            }
        }
        return ans;
    }

}
