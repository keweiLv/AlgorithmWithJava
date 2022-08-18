package LC_StudyPlan75_Lv2;

/**
 * @author Kezi
 * @date 2022年08月18日 22:13
 */
public class Solution {

	/**
	 * 快乐数
	 * 如果 n 是一个快乐数，即没有循环，那么快跑者最终会比慢跑者先到达数字 1。
	 * 如果 n 不是一个快乐的数字，那么最终快跑者和慢跑者将在同一个数字上相遇。
	 */
	public boolean isHappy(int n) {
		int slow = n, fast = squareSum(n);
		while (slow != fast) {
			slow = squareSum(slow);
			fast = squareSum(squareSum(fast));
		}
		return slow == 1;
	}
	private int squareSum(int n) {
		int sum = 0;
		while (n > 0) {
			int digit = n % 10;
			sum += digit * digit;
			n /= 10;
		}
		return sum;
	}
}
