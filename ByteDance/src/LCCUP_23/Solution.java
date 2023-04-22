package LCCUP_23;

import java.util.*;

public class Solution {

    public static void main(String[] args) {
        int[][] rec = new int[3][2];
        System.out.println(Arrays.deepToString(rec));
    }

    // 补给马车
    public static int[] supplyWagon(int[] supplies) {
        int bn = supplies.length;
        int n = bn / 2;
        int start = 0, end = 0;
        while (supplies.length != n) {
            int mx = Integer.MAX_VALUE;
            for (int i = 0; i < supplies.length - 1; i++) {
                if (supplies[i] + supplies[i + 1] < mx) {
                    mx = supplies[i] + supplies[i + 1];
                    start = i;
                    end = i + 1;
                }
            }
            supplies[start] = supplies[start] + supplies[end];
            while (end < supplies.length - 1) {
                supplies[end] = supplies[end + 1];
                end++;
            }
            supplies = Arrays.copyOfRange(supplies, 0, supplies.length - 1);
        }
        return supplies;
    }

    // 探险营地
    public static int adventureCamp(String[] expeditions) {
        int res = -1;
        String[] all = expeditions[0].split("->");
        Set<String> set = new HashSet<>(Arrays.asList(all));
        int mx = 0;
        for (int i = 1; i < expeditions.length; i++) {
            int cnt = 0;
            String[] split = expeditions[i].split("->");
            if (split.length < 2) {
                if (Objects.equals(expeditions[i], "")) {
                    continue;
                }
                if (!set.contains(expeditions[i])) {
                    cnt++;
                    set.add(expeditions[i]);
                }
                if (cnt > mx) {
                    mx = cnt;
                    res = i;
                }
            } else {
                for (String s : split) {
                    if (!set.contains(s)) {
                        cnt++;
                        set.add(s);
                    }
                }
                if (cnt > mx) {
                    mx = cnt;
                    res = i;
                }
            }

        }
        return res;
    }
}

