package LC_RACE.WeeklyRace_343;

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

}
