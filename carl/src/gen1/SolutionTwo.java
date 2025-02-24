package carl.src.gen1;

import java.util.ArrayList;
import java.util.List;

public class SolutionTwo {

//    public static void main(String[] args) {
//        String s = "BBABBA";
//        int i = minDeletionsToObtainStringInRightFormat(s);
//        System.out.println(i);

//        int i = minDeletions(s);
//        System.out.println(i);

//    }


    private static final List<byte[]> memoryLeakList = new ArrayList<>();

    public static void main(String[] args) {

    }


    public static int minDeletionsToObtainStringInRightFormat(String s) {
        // write code here
        int a = s.lastIndexOf("A");
        if (a == -1) {
            return 0;
        }
        int countB = 0;
        for (int i = 0; i < a; i++) {
            if (s.charAt(i) == 'B') {
                countB++;
            }
        }
        return Math.min(countB, a-1);
    }

    public static String stringCut(String val) {
        // write code here
        int cnt = 0;
        String res = "";
        for (int i = 0; i < val.length(); i++) {
            if (val.charAt(i) > 127) {
                cnt += 3;
            } else {
                cnt++;
            }
            if (cnt > 10) {
                return res;
            }
            res = val.substring(0, i + 1);
        }
        return "";


//        int length = val.getBytes().length;
//        for (int i = 0; i < val.length(); i++) {
//
//            int n = val.substring(0, i).getBytes().length;
//            if (n >= 10){
//                return val.substring(0, i);
//            }
//
//        }
//        return "";
    }

    public static int minDeletions(String s) {
        int n = s.length();
        int dp = 0;  // dp[i] 的值
        int countB = 0;  // 记录前面出现的 B 的数量

        for (char c : s.toCharArray()) {
            if (c == 'B') {
                countB++;  // 统计遇到的 B 的数量
            } else { // c == 'A'
                // 要么删除当前 A（即 dp[i-1] + 1），要么删除前面的所有 B（即 countB）
                dp = Math.min(dp + 1, countB);
            }
        }

        return dp;
    }

    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 最长山脉的长度
     * @param nums int整型一维数组 每个元素表示一座山的高度
     * @return int整型
     */
    public static int maxLength (int[] nums) {
        // write code here
        int n = nums.length;
        if (n < 3) {
            return 0;
        }
        int res = 0;
        int left = 0;
        int right = 0;
        while (left < n - 2) {
            if (nums[left] < nums[left + 1]) {
                right = left + 1;
                while (right < n - 1 && nums[right] < nums[right + 1]) {
                    right++;
                }
                if (right < n - 1 && nums[right] > nums[right + 1]) {
                    while (right < n - 1 && nums[right] > nums[right + 1]) {
                        right++;
                    }
                    res = Math.max(res, right - left + 1);
                }
                left = right;
            } else {
                left++;
            }
        }
        return res;
    }
}
