package LC_StudyPlan75_Lv3;

/**
 * @author Kezi
 * @date 2022年09月08日 0:02
 */
public class Solution {

	/**
	 * 位1的个数
	 * 算术右移 >> ：舍弃最低位，高位用符号位填补；
	 * 逻辑右移 >>> ：舍弃最低位，高位用 0 填补
	 * 对于负数而言，其二进制最高位是 1，如果使用算术右移，那么高位填补的仍然是 1。也就是 n 永远不会为 0
	 */
	public int hammingWeight(int n) {
		int count = 0;
		while (n!=0){
			count += n & 1;
			n >>>= 1;
		}
		return count;
	}
}
