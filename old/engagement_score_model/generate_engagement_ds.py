import numpy as np
import pandas as pd
import os
import pdb

########################################
# Assumes the first col of df is name
# the rest of the cols are user group engagements
#######################################
def get_engagement_category_thres( df ):
    engagements = []
    for col in df.columns[1:]:
        engagements += df[col].values.tolist()
    mean = np.mean( engagements )
    std = np.std( engagements )
    categoryThres = [-999999, mean-0.3*std, mean+0.3+std, 999999]
    print( 'Engagement score breakdown is %s' 
        % categoryThres )
    return categoryThres

def convert_to_categorical( thresholds, df ):
    catDf = df.copy()
    for col in df.columns[1:]:
        for i in range(1,len(thresholds)):
            idx = (df[col] < thresholds[i]) & (
                    df[col] >= thresholds[i-1])
            print( 'Col %s between %.1f and %.1f, there are '
                    '%s datapoints' % ( col, thresholds[i],
                        thresholds[i-1], np.sum( idx ) ) )
            catDf.loc[idx,col] = i-1
    return catDf

###################################
#This function constructs the engagement score dataset
#tweetFile is the raw csv file from step 1
#userFile is the user group assignment from step 2
##################################
def construct_engagement_score_ds( tweetFiles, userFile ):
    for tFile in tweetFiles:
        assert os.path.isfile(tFile), 'Invalid Tweet file'
    assert os.path.isfile(userFile), 'Invalid userFile'

    userNameCol = 'screen_name'
    parentIdCol = 'parent_id'
    usrGrpCol = 'group'
    idCol = 'id'
    egmtCol = 'engagement'

    tweetDf = None
    for tFile in tweetFiles:
        curDf = pd.read_csv( tFile )
        if tweetDf is None:
            tweetDf = curDf
        else:
            tweetDf = tweetDf.append( curDf )
        print( tweetDf.shape )
    userDf = pd.read_csv( userFile )

    userGrps = userDf[usrGrpCol].unique()
    print( 'User groups: %s' % userGrps )
    pdb.set_trace()
    #Add user group assignment to tweet Df
    tweetDf = pd.merge( tweetDf, userDf, how='inner', on=userNameCol )
    print( 'tweetDf: ', tweetDf.shape )

    engagementDf = []
    grpDetails = []

    for pId in tweetDf[parentIdCol].dropna().unique().astype(
            tweetDf[idCol].dtype ):
        curTweetEntry = []
        
        tweet = tweetDf[tweetDf[idCol] == pId]
        if tweet.shape[0] == 0:
            continue

        parentFollower = tweet['follower_count'].values[0]+1
        curTweetEntry.append( tweet['clean_text'].values[0], parentFollower ) 
        replyDf = tweetDf[tweetDf[parentIdCol]==pId]
        # Parent follower cnt makes the distribution somewhat unpredicatble
        replyDf.loc[:,egmtCol] = np.log(( replyDf['favorite_count'] +
                replyDf['retweet_count'] + 1) * ( replyDf['polarity']+1.0001 ))
        #replyDf.loc[:,egmtCol] = np.log((( replyDf['favorite_count'] +
        #        replyDf['retweet_count'] + 1) * ( replyDf['polarity'] + 1 ) / 
        #        parentFollower ).values[0] + 1E-15)
        curTweetDetail = [ pId ]
        for grp in sorted(userGrps):
            if grp in replyDf[usrGrpCol].unique():
                egmts = replyDf[replyDf[usrGrpCol]==grp][egmtCol]
                curTweetEntry.append( egmts.mean() )
                curTweetDetail.append( egmts.values )
            else:
                curTweetDetail.append( [] )
                curTweetEntry.append( 0 )
        pdb.set_trace()
        grpDetails.append( curTweetDetail )
        engagementDf.append( curTweetEntry )
        if( len( engagementDf ) % 100 ) == 0:
            print( 'Recorded %s entries' % len( engagementDf ) )
            break

    engagementDf = pd.DataFrame( engagementDf )
    engagementDf.columns = ['text'] + sorted( userGrps ) 
    curTweetDetail = pd.DataFrame( curTweetDetail )
    curTweetEntry.to_csv( 'backup.csv', index=False )
    return engagementDf

tweetFiles = [ '../data/GWUFile1.csv',
               '../data/GWUFile2.csv',
               '../data/GWUFile3.csv',
               '../data/GWUFile4.csv',
            ]
userFile = '../clustering/clusters.csv'
oFile = './engagement.csv'
trainFile = './train.csv'
valFile = './val.csv'
testFile = './test.csv'

testRatio = 0.1
valRatio = 0.1

engagementDf = construct_engagement_score_ds(tweetFiles, userFile)
engagementDf.to_csv('harold.csv',index=False)
pdb.set_trace()
engagementDf = pd.read_csv('harold.csv')
engagementDf = engagementDf[ (engagementDf['1'] != 0) & (engagementDf['0']!=0)]
thresholds = get_engagement_category_thres( engagementDf )
engagementDf = convert_to_categorical( thresholds, engagementDf )
engagementDf.to_csv(oFile, index=False)

testSize = np.round( engagementDf.shape[0] * testRatio, 0 )
valSize = np.round( engagementDf.shape[0] * valRatio, 0 )
print( 'Test size is %s, val size is %s' % ( testSize, valSize ) )
engagementDf.loc[0:testSize].to_csv( testFile, index=False )
engagementDf.loc[testSize:(testSize+valSize)].to_csv( valFile, index=False )
engagementDf.loc[(testSize+valSize):].to_csv( trainFile, index=False )
