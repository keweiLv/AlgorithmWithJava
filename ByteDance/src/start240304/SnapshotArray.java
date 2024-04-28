package start240304;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SnapshotArray {
    private int curSnapId;

    private final Map<Integer, List<int[]>> history = new HashMap<>();

    public SnapshotArray(int length) {

    }

    public void set(int index, int val) {
        history.computeIfAbsent(index, k -> new ArrayList<>()).add(new int[]{curSnapId, val});
    }

    public int snap() {
        return curSnapId++;
    }

    public int get(int index, int snap_id) {
        if (!history.containsKey(index)) {
            return 0;
        }
        List<int[]> ints = history.get(index);
        int j = search(ints, snap_id);
        return j < 0 ? 0 : ints.get(j)[1];
    }

    private int search(List<int[]> ints, int snapId) {
        int left = -1;
        int right = ints.size();
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (ints.get(mid)[0] <= snapId) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return left;
    }

    // 负二进制转换
    public String baseNeg2(int n) {
        if (n == 0) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        while (n != 0) {
            int mod = n % (-2);
            n = n / (-2);
            if (mod == -1) {
                n++;
                mod = 1;
            }
            sb.append(mod);
        }
        return sb.reverse().toString();
    }
}
