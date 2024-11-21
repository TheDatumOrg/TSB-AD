const COLORS = {
    white:   '#ffffff',
    black:   '#000000',
    green:   '#49F2CC',
    lightGrey: '#777',
    grey:    '#29363B',
    cyan:    'cyan',
    yellow:  '#FFE202',
    hotpink: 'deeppink',
  };
  
  // ADD CUSTOM SHAPES
  class D extends mojs.CustomShape {
    getShape () {
      return '<path d="M.3105 97.7669V.7469h23.7796c16.7165 0 29.8045 4.3271 39.2737 12.9813 9.4692 8.6542 14.2037 20.5003 14.2037 35.5287 0 9.0132-1.8725 17.0076-5.6175 23.9833-3.745 6.9854-9.246 12.729-16.4837 17.2308-4.1234 2.5904-8.596 4.4435-13.4276 5.5884-4.8413 1.1351-10.8177 1.7076-17.939 1.7076h-23.7796Zm21.1213-20.8108h4.6958c9.343 0 16.6583-2.4546 21.9265-7.3638 5.2682-4.9092 7.9071-11.6909 7.9071-20.3354s-2.6389-15.4553-7.9071-20.4033c-5.2682-4.948-12.5738-7.4317-21.9265-7.4317h-4.6958v55.5342Z"></path>';
    }
  }
  mojs.addShape('d', D);
  
  class A extends mojs.CustomShape {
    getShape () {
      return '<path d="M0 98 48 0H48L96 98H72.912L67 84H29L22 98H0ZM35 68H60L48 38 35 68Z"></path>';
    }
  }
  mojs.addShape('a', A);
  
  class T extends mojs.CustomShape {
    getShape () {
      return '<path d="M1.1254 1.1742 58.7456 1.1742 58.7456 17.6773 40.0305 17.6773 39.5454 98.1942 20.8302 98.1942 20.0928 18.1624 1.1254 18.4147 1.1254 1.1742z"></path>';
    }
  }
  mojs.addShape('t', T);
  
  class U extends mojs.CustomShape {
    getShape () {
      return '<path d="M21.1116 1.1552v51.6534c0 8.2952 1.4456 14.5045 4.3465 18.6278 2.9009 4.1234 7.2862 6.1899 13.1559 6.1899s9.9931-1.921 12.9716-5.7727c2.9785-3.842 4.4726-9.44 4.4726-16.7845V1.1552h19.045v53.9237c0 13.5634-3.3084 24.1386-9.9251 31.7255-6.6168 7.587-15.7949 11.3707-27.5537 11.3707s-21.1116-3.7547-27.7089-11.2737c-6.5974-7.5191-9.896-18.0845-9.896-31.6867V1.1552h21.0824Z"></path>';
    }
  }
  mojs.addShape('u', U);
  
  class M extends mojs.CustomShape {
    getShape () {
      return '<path d="M.5.5.277 97.8288 17.4074 97.8288 17.5 47.5 45.5 77.5 74.5 47.5 74.1788 97.8288 91.9364 97.8288 91.5.5 45.5 50.5.5.5z"></path>';
    }
  }
  mojs.addShape('m', M);
  
  // VARIABLES
  const {approximate} = mojs.easing,
        shiftCurve = approximate(mojs.easing.path( 'M0,100 C50,100 50,100 50,50 C50,0 50,0 100,0' )),
        scaleCurve = approximate(mojs.easing.path( 'M0,100 C21.3776817,95.8051376 50,77.3262711 50,-700 C50,80.1708527 76.6222458,93.9449005 100,100' )),
        charSize = 25,
        leftStep = 22,
        logo     = document.querySelector('#js-logo');
  
  const CHAR_OPTS = {
          parent:       logo,
          isForce3d:    true,
          fill:         'white',
          radius:       charSize/2,
          stroke:       'white',
          strokeWidth:  0
        }
  
  const CHAR_HIDE_THEN = {
          delay: 930,
          isShowEnd: false
        }
  
  // HELPERS
  const scale = function (curve, n) {
    return (p) => { return n*curve(p); }
  }
  const increase = function (curve, n) {
    return (p) => { return n + curve(p); }
  }
  
  // CURVES
  const scaleC = approximate( increase( scaleCurve, 1 ) ),
        scaledCurve = ( amount ) => {
          return increase( scale( scaleCurve, amount ), 1 );
        },
        scaleCShort = approximate( scaledCurve(.75) );
  
  
  // SHAPES
  const dCharacter = new mojs.Shape({
      ...CHAR_OPTS,
      shape:    'd',
      left:     2.0*leftStep + '%',
      x:        -7,
      y:        { [350] : 150, easing: shiftCurve },
      scaleY:   { 1 : 1, curve: scaleCShort },
      origin:   { '50% 100%' : '50% 0%', easing: shiftCurve },
      delay:    232,
      duration: 680,
    }).then({
      delay:   115,
      y:       { to: 0, easing: shiftCurve },
      scaleY:  { 1: 1, curve: approximate( scaledCurve(.5) ) },
      origin: { '50% 100%' : '50% 0%', easing: shiftCurve }
    });
  
  const aCharacter = new mojs.Shape({
      ...CHAR_OPTS, 
      shape:        'a',
      left:         2.45*leftStep + '%',
      delay:        350,
      duration:     465,
      strokeWidth:  0,
      x:            200,
      y:            { [-100] : 250, easing: shiftCurve },
      scaleY:       { 1: 1, curve: scaleC },
      origin:       { '50% 0%' : '50% 100%', easing: shiftCurve }
    }).then({
      duration:     407,
      x:            { to: 0, easing: shiftCurve },
      scaleX:       { 1: 1, curve: scaleCShort },
      origin:       { '100% 50%' : '0% 50%', easing: shiftCurve }
    }).then({
      duration:     700,
      y:            { to: 0, easing: shiftCurve },
      scaleY:       { 1: 1, curve: scaleCShort },
      origin:       { '50% 100%' : '50% 0%', easing: shiftCurve }
    });
  
  const tCharacter = new mojs.Shape({
      ...CHAR_OPTS,
      shape: 't',
      left:       3.3*leftStep + '%',
      y:            { 0: 35 },
      delay:        1337,
      x:         0,
      y:          { [250] : -200, easing: shiftCurve 
      },
      easing:       'expo.out',
      isForce3d:    true,
      isShowStart:  false,
  }).then({
      y:            0,
      easing:       'elastic.out',
      duration:     1160,
  });
  
  const uCharacter = new mojs.Shape({
        ...CHAR_OPTS,
        shape:      'u',
        left:       3.925*leftStep + '%',
        delay:      40,
        duration:   580,
        x:         -200,
        y:          { [250] : -100, easing: shiftCurve },
        scaleY:     { 1: 1, curve: scaleC },
        origin:     { '50% 100%' : '50% 0%', easing: shiftCurve }
      })
  .then({
        duration:   523,
        x:          { to: 0, easing: shiftCurve },
        scaleX:     { 1: 1, curve: scaleCShort },
        origin:     { '0% 50%' : '100% 50%', easing: shiftCurve }
      })
  .then({
        y:          { to: 0, easing: shiftCurve },
        // x:          { to: charSize, easing: shiftCurve },
        scaleY:     { 1: 1, curve:  approximate( scaledCurve(.5) ) },
        origin:     { '50% 0%' : '50% 100%', easing: shiftCurve }
  });
  
  const mCharacter = new mojs.Shape({
        ...CHAR_OPTS,
        shape:      'm',
        left:       4.69*leftStep + '%',
        delay:      116,
        duration:   523,
        x:          { 500: 0, easing: shiftCurve },
        y:          200,
        scaleX:     { 1: 1, curve: scaleC },
        origin:     { '100% 50%' : '0% 100%', easing: shiftCurve }
      })
  .then({
        delay:      116,
        y:          { to: 0, easing: shiftCurve },
        scaleY:     { 1: 1, curve: scaleCShort },
        origin:     { '50% 100%' : '50% 0%', easing: shiftCurve }
      });
  
  // LINES
  
  let LINE_OPTS = {
    shape:        'line',
    strokeWidth:  { 10: 0 },
    stroke:       COLORS.cyan,
    radius:       44,
    parent:       logo,
    angle:        90,
    duration:     465,
    delay:        495,
    radiusY:      0,
    strokeDasharray: '100% 100%',
    strokeDashoffset: { '100%': '-100%' }
  };
  
  let line1 = new mojs.Shape({
    ...LINE_OPTS,
    x:  189,
    y:  { [-20] : 160 },
  });
  
  let line2 = new mojs.Shape({
    ...LINE_OPTS,
    x: -175,
    y: { 200 : -20 },
    stroke: COLORS.hotpink,
    strokeDashoffset: { '-100%' : '100%' },
    delay: 290
  });
  
  let line3 = new mojs.Shape({
    ...LINE_OPTS,
    radius: 53,
    y: 30,
    stroke: COLORS.yellow,
    strokeDashoffset: { '-100%': '100%' },
    delay: 804,
    angle: 0
  });
  
  let StaggerShape = new mojs.stagger( mojs.Shape );
  
let underlines = new StaggerShape({
    ...LINE_OPTS,
    radius: 60,
    angle: 0,
    radiusY: 0,
    y: 20,
    strokeWidth: 2,
    stroke: [ COLORS.hotpink, COLORS.yellow, COLORS.cyan, COLORS.white ],
    duration: 1000,
    delay: 'stagger(1999, 145)',
    strokeDasharray: null,
    strokeDashoffset: null,
    scaleX: { 0: 1 },    
    origin: '0 50%',
    quantifier: 'stroke',
    easing: 'expo.out',
    x: 25,
    isForce3d: true
});
  
  // SHAPES
  let shapes = new StaggerShape({
    parent:       logo,
    left:         '100%',
    x:            [ 15, 45, 35 ],
    y:            [ -25, -5, -35 ],
    quantifier:   'shape',
    shape:        [ 'circle', 'polygon', 'rect' ],
    radius:       7,
    fill:         'none',
    stroke:       [ 'deeppink', COLORS.cyan, COLORS.yellow ],
    strokeWidth:  { 5 : 0 },
    scale:        { .75 : 1 },
    delay:        'stagger(2249, 58)',
    isTimelineLess: true
  });
  
//   // LOGO SHIFT
//   let yShift = 0;
//   const getYShift = () => {
//     const w = window;
//     const height = w.innerHeight || e.clientHeight || g.clientHeight;
//     yShift = height/1.5;
//   }
  
//   getYShift();
//   window.addEventListener('resize', getYShift);
  
  
//   let logoShift = new mojs.Tween({
//     duration: 349,
//     delay:    2999,
//     onUpdate (p) {
//       var shiftP = mojs.easing.cubic.in( p );
//       var scaleP = mojs.easing.quad.in( p );
      
//       mojs.h.setPrefixedStyle( logo, 'transform',
//         `translate(0px, ${ yShift*shiftP }px)
//         scaleY(${ 1 + 25*scaleP })`
//       );
//     }
//   })
  
  const timeline = new mojs.Timeline({
    onStart () {
      mojs.h.setPrefixedStyle( logo, 'transform', 'scale(3)' );
      
    },
    onComplete () {
      let fadeOut = new mojs.Tween({
        duration: 1000, 
        onUpdate: (progress) => {
          let opacity = 1 - progress;
          let modules = underlines.childModules;
          for (let i = 0; i < modules.length; i++) {
            modules[i].el.style.opacity = opacity;
          }
        }
      });
      fadeOut.play();
    },
  });
  timeline
    .add(
    dCharacter, aCharacter, tCharacter, uCharacter, mCharacter,
    underlines, line1, line2, line3,
    shapes
  );
  
//   new MojsPlayer({ add: timeline, isRepeat:true, isPlaying:true});
  timeline.play()

  