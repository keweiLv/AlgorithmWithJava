package prepareForByte;

/**
 * @author Kezi
 * @date 2023年02月18日 0:10
 */
public class Solution {


	// 买卖股票的最佳时机
	public int maxProfit(int[] prices) {
		int minPrices = Integer.MAX_VALUE;
		int maxProfit = 0;
		for (int i = 0;i<prices.length;i++){
			if (prices[i] < minPrices){
				minPrices = prices[i];
			}else if (prices[i] - minPrices > maxProfit){
				maxProfit = prices[i] - minPrices;
			}
		}
		return maxProfit;
	}
}
