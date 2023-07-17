package again230717;

/**
 * 2023-07-17 another again
 */
public class Solution {


    public static void main(String[] args) {
        String num1 = "11", num2 = "123";
        System.out.println(addStrings(num1, num2));
    }

    // 字符串相加
    public static String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int m = num1.length() - 1, n = num2.length() - 1, carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            int sum = (m >= 0 ? num1.charAt(m) - '0' : 0) + (n >= 0 ? num2.charAt(n) - '0' : 0) + carry;
            carry = sum / 10;
            sum = sum % 10;
            sb.append(sum);
            m--;
            n--;
        }
        return sb.reverse().toString();
    }

    // 分割等和子集
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if ((sum & 1) == 1) {
            return false;
        }
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            for (int j = target; j >= num; j--) {
                dp[j] |= dp[j - num];
            }
        }
        return dp[target];
    }
}
