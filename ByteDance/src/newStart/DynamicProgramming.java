package newStart;

public class DynamicProgramming {

    // 爬楼梯
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int p = 1, q = 2, r = 0;
        for (int i = 3; i <= n; i++) {
            r = p + q;
            p = q;
            q = r;
        }
        return r;
    }

    // 斐波那契数
    public int fib(int n) {
        if (n <= 1) {
            return n;
        }
        int[] f = new int[n + 1];
        f[0] = 0;
        f[1] = 1;
        for (int i = 2; i <= n; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }

    // 第N个泰波那契数
    public int tribonacci(int n) {
        if(n == 0){
            return 0;
        }
        if(n == 1 || n == 2){
            return 1;
        }
        int[] f = new int[n+1];
        f[0] = 0;
        f[1] = 1;
        f[2] = 1;
        for (int i = 3; i <= n ; i++) {
            f[i] = f[i-1] + f[i-2] + f[i-3];
        }
        return f[n];
    }

    // 删除并获得点数
    int[] cnts = new int[10010];
    public int deleteAndEarn(int[] nums) {
        int n = nums.length;
        int max = 0;
        for (int num:nums){
            cnts[num]++;
            max = Math.max(max,num);
        }
        int[][] f = new int[max+1][2];
        for (int i = 1;i<=max;i++){
            f[i][0] = Math.max(f[i-1][0],f[i-1][1]);
            f[i][1] = f[i-1][0] + i * cnts[i];
        }
        return Math.max(f[max][0],f[max][1]);
    }
}
