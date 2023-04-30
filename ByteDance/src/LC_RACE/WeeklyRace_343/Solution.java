package LC_RACE.WeeklyRace_343;

import java.util.HashMap;
import java.util.Map;

/**
 * 第 343 场周赛
 */
public class Solution {

    // 保龄球游戏的获胜者
    public int isWinner(int[] player1, int[] player2) {
        if (player1.length == 0) {
            return 0;
        }
        int pl1 = player1[0], pl2 = player2[0];
        int n = player1.length;
        for (int i = 1; i < n; i++) {
            if (i - 2 >= 0) {
                pl1 += player1[i - 1] == 10 || player1[i - 2] == 10 ? 2 * player1[i] : player1[i];
                pl2 += player2[i - 1] == 10 || player2[i - 2] == 10 ? 2 * player2[i] : player2[i];
            } else {
                pl1 += player1[i - 1] == 10 ? 2 * player1[i] : player1[i];
                pl2 += player2[i - 1] == 10 ? 2 * player2[i] : player2[i];
            }
        }
        if (pl1 == pl2) {
            return 0;
        }
        return pl1 > pl2 ? 1 : 2;
    }

    // 找出叠涂元素
    public int firstCompleteIndex(int[] arr, int[][] mat) {
        int n = mat.length, m = mat[0].length;
        Map<Integer, int[]> map = new HashMap<>();
        int[] rows = new int[n];
        int[] cols = new int[m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                map.put(mat[i][j], new int[]{i, j});
            }
        }
        for (int i = 0; i < arr.length; i++) {
            int[] ints = map.get(arr[i]);
            rows[ints[0]]++;
            cols[ints[1]]++;
            if (rows[ints[0]] == m || cols[ints[1]] == n){
                return i;
            }
        }
        return -1;
    }
}
